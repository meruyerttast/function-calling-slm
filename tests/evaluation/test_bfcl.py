# tests/evaluation/test_bfcl.py
import json
import subprocess
from pathlib import Path

import pytest

from src.evaluation.bfcl import (
    _coerce,
    aggregate_weighted_score,
    build_evaluate_cmd,
    build_generate_cmd,
    load_config,
    merge_with_existing_scores,
    parse_bfcl_score_output,
    run_eval,
    shim_bfcl_output,
)


# Mirror of upstream's COLUMNS_OVERALL (constants/column_headers.py:62)
# plus the header row of data_non_live.csv (COLUMNS_NON_LIVE).
REAL_OVERALL_HEADER = (
    "Rank,Overall Acc,Model,Model Link,Total Cost ($),"
    "Latency Mean (s),Latency Standard Deviation (s),Latency 95th Percentile (s),"
    "Non-Live AST Acc,Non-Live Simple AST,Non-Live Multiple AST,"
    "Non-Live Parallel AST,Non-Live Parallel Multiple AST,"
    "Live Acc,Live Simple AST,Live Multiple AST,Live Parallel AST,Live Parallel Multiple AST,"
    "Multi Turn Acc,Multi Turn Base,Multi Turn Miss Func,Multi Turn Miss Param,Multi Turn Long Context,"
    "Web Search Acc,Web Search Base,Web Search No Snippet,"
    "Memory Acc,Memory KV,Memory Vector,Memory Recursive Summarization,"
    "Relevance Detection,Irrelevance Detection,"
    "Format Sensitivity Max Delta,Format Sensitivity Standard Deviation,"
    "Organization,License"
)

REAL_NON_LIVE_HEADER = (
    "Rank,Model,Non-Live Overall Acc,AST Summary,Simple AST,"
    "Python Simple AST,Java Simple AST,JavaScript Simple AST,"
    "Multiple AST,Parallel AST,Parallel Multiple AST,Irrelevance Detection"
)


@pytest.fixture
def base_config():
    return {
        "model": {
            "name": "Qwen/Qwen2.5-3B-Instruct",
            "bfcl_model_id": "Qwen/Qwen2.5-3B-Instruct-FC",
            "adapter_path": None,
        },
        "lora": {"enable": False, "max_rank": 64},
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "backend": "vllm",
        },
        "bfcl": {
            "version_pin": "2026.3.23",
            "test_categories": ["all"],
            "result_dir_template": "results/{variant}/generation",
            "score_dir_template": "results/{variant}",
            "limit": None,
        },
        "weights": {
            "agentic": 0.4,
            "multi_turn": 0.3,
            "live": 0.1,
            "non_live": 0.1,
            "hallucination": 0.1,
        },
        "logging": {"report_to": None, "wandb_project": "info290-function-calling-slm"},
        "output_dir": "results",
    }


class TestAggregateWeightedScore:
    def test_exact_sum(self):
        scores = {"agentic": 0.5, "multi_turn": 0.6, "live": 0.7, "non_live": 0.8, "hallucination": 0.9}
        weights = {"agentic": 0.4, "multi_turn": 0.3, "live": 0.1, "non_live": 0.1, "hallucination": 0.1}
        # 0.5*0.4 + 0.6*0.3 + 0.7*0.1 + 0.8*0.1 + 0.9*0.1 = 0.2 + 0.18 + 0.07 + 0.08 + 0.09 = 0.62
        assert abs(aggregate_weighted_score(scores, weights) - 0.62) < 1e-9

    def test_zero_scores(self):
        scores = {"agentic": 0, "multi_turn": 0, "live": 0, "non_live": 0, "hallucination": 0}
        weights = {"agentic": 0.4, "multi_turn": 0.3, "live": 0.1, "non_live": 0.1, "hallucination": 0.1}
        assert aggregate_weighted_score(scores, weights) == 0.0

    def test_missing_bucket_treated_as_zero(self):
        # When only agentic was scored, other buckets should be treated as 0 in the overall.
        scores = {"agentic": 1.0}
        weights = {"agentic": 0.4, "multi_turn": 0.3, "live": 0.1, "non_live": 0.1, "hallucination": 0.1}
        assert aggregate_weighted_score(scores, weights) == 0.4

    def test_none_bucket_treated_as_zero(self):
        scores = {"agentic": None, "multi_turn": 0.5, "live": None, "non_live": None, "hallucination": None}
        weights = {"agentic": 0.4, "multi_turn": 0.3, "live": 0.1, "non_live": 0.1, "hallucination": 0.1}
        assert abs(aggregate_weighted_score(scores, weights) - 0.15) < 1e-9


class TestBuildGenerateCmd:
    def test_base_model_has_no_lora_flags(self, base_config):
        cmd = build_generate_cmd(base_config, variant="v1", adapter=None, test_category="all")
        assert cmd[:2] == ["bfcl", "generate"]
        assert "Qwen/Qwen2.5-3B-Instruct-FC" in cmd
        assert "--backend" in cmd and "vllm" in cmd
        assert "--temperature" in cmd
        assert "--enable-lora" not in cmd
        assert "--lora-modules" not in cmd
        assert "--max-lora-rank" not in cmd

    def test_adapter_emits_lora_flags(self, base_config, tmp_path):
        adapter_dir = tmp_path / "sft-v2"
        adapter_dir.mkdir()
        cmd = build_generate_cmd(base_config, variant="v2", adapter=adapter_dir, test_category="all")
        assert "--enable-lora" in cmd
        assert "--max-lora-rank" in cmd
        assert "64" in cmd
        assert "--lora-modules" in cmd
        idx = cmd.index("--lora-modules")
        lora_arg = cmd[idx + 1]
        # Format is name="/abs/path"
        assert lora_arg.startswith("v2=")
        assert str(adapter_dir.resolve()) in lora_arg

    def test_test_category_all(self, base_config):
        cmd = build_generate_cmd(base_config, variant="v1", test_category="all")
        assert "--test-category" in cmd
        idx = cmd.index("--test-category")
        assert cmd[idx + 1] == "all"

    def test_test_category_non_live(self, base_config):
        cmd = build_generate_cmd(base_config, variant="v1", test_category="non_live")
        idx = cmd.index("--test-category")
        assert cmd[idx + 1] == "non_live"

    def test_result_dir_template_interpolates_variant(self, base_config):
        cmd = build_generate_cmd(base_config, variant="v2", test_category="all")
        assert "--result-dir" in cmd
        idx = cmd.index("--result-dir")
        assert "v2" in cmd[idx + 1]
        assert "generation" in cmd[idx + 1]

    def test_temperature_is_generation_config_value(self, base_config):
        cmd = build_generate_cmd(base_config, variant="v1", test_category="all")
        idx = cmd.index("--temperature")
        assert cmd[idx + 1] == "0.0"


class TestBuildEvaluateCmd:
    def test_basic(self, base_config):
        cmd = build_evaluate_cmd(base_config, variant="v1", test_category="all")
        assert cmd[:2] == ["bfcl", "evaluate"]
        assert "Qwen/Qwen2.5-3B-Instruct-FC" in cmd
        assert "--result-dir" in cmd
        assert "--score-dir" in cmd
        idx = cmd.index("--test-category")
        assert cmd[idx + 1] == "all"

    def test_score_dir_is_variant_dir(self, base_config):
        cmd = build_evaluate_cmd(base_config, variant="v3", test_category="all")
        idx = cmd.index("--score-dir")
        assert "v3" in cmd[idx + 1]


class TestShimBfclOutput:
    def test_full_canonical_shape(self, base_config):
        # Upstream "raw" shape mirrors the aggregation in eval_runner_helper.py:509-519:
        # [non_live, live, total_irrelevance, multi_turn, agentic] with weights [10,10,10,30,40].
        # We rename total_irrelevance -> hallucination in the canonical schema.
        raw = {
            "overall_accuracy_non_live": 0.55,
            "overall_accuracy_live": 0.45,
            "total_irrelevance": 0.30,
            "overall_accuracy_multi_turn": 0.10,
            "overall_accuracy_agentic": 0.05,
            "format_sensitivity": {
                "accuracy_max_delta": 0.12,
                "accuracy_std": 0.04,
            },
        }
        canonical = shim_bfcl_output(raw, base_config, variant="v1", adapter=None)
        assert canonical["variant"] == "v1"
        assert canonical["model"] == "Qwen/Qwen2.5-3B-Instruct"
        assert canonical["adapter"] is None
        assert canonical["bfcl_version"] == "2026.3.23"
        assert canonical["top_level"]["non_live"] == 0.55
        assert canonical["top_level"]["live"] == 0.45
        assert canonical["top_level"]["hallucination"] == 0.30
        assert canonical["top_level"]["multi_turn"] == 0.10
        assert canonical["top_level"]["agentic"] == 0.05
        assert canonical["weights"] == base_config["weights"]
        # 0.05*0.4 + 0.10*0.3 + 0.45*0.1 + 0.55*0.1 + 0.30*0.1
        # = 0.02 + 0.03 + 0.045 + 0.055 + 0.03 = 0.18
        assert abs(canonical["overall"] - 0.18) < 1e-9
        assert canonical["side_metrics"]["format_sensitivity_std"] == 0.04
        assert canonical["side_metrics"]["format_sensitivity_max_delta"] == 0.12
        assert canonical["generation_config"]["temperature"] == 0.0

    def test_missing_bucket_is_null_not_omitted(self, base_config):
        # When --test-category was non_live, only non_live is scored.
        raw = {"overall_accuracy_non_live": 0.55}
        canonical = shim_bfcl_output(raw, base_config, variant="v1", adapter=None)
        assert canonical["top_level"]["non_live"] == 0.55
        for bucket in ("live", "multi_turn", "agentic", "hallucination"):
            assert bucket in canonical["top_level"]
            assert canonical["top_level"][bucket] is None

    def test_adapter_path_recorded(self, base_config, tmp_path):
        adapter_dir = tmp_path / "sft-v2" / "final"
        adapter_dir.mkdir(parents=True)
        raw = {"overall_accuracy_non_live": 0.6}
        canonical = shim_bfcl_output(raw, base_config, variant="v2", adapter=adapter_dir)
        assert canonical["adapter"] == str(adapter_dir)

    def test_non_live_run_masks_unevaluated_buckets_as_null(self, base_config):
        # bfcl's data_overall.csv reports Live Acc = 0% when no live tests ran,
        # which _coerce turns into 0.0. The shim must mask those buckets as null
        # since they weren't actually evaluated, not scored zero.
        raw = {
            "overall_accuracy_non_live": 0.80,
            "overall_accuracy_live": 0.0,
            "overall_accuracy_multi_turn": 0.0,
            "total_irrelevance": None,
            "overall_accuracy_agentic": None,
        }
        canonical = shim_bfcl_output(
            raw, base_config, variant="v1", adapter=None, test_category="non_live"
        )
        assert canonical["top_level"]["non_live"] == 0.80
        for bucket in ("live", "multi_turn", "hallucination", "agentic"):
            assert canonical["top_level"][bucket] is None, (
                f"{bucket} must be null when --test-category=non_live, got "
                f"{canonical['top_level'][bucket]}"
            )


class TestRunEval:
    def test_missing_adapter_raises_before_subprocess(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(FileNotFoundError):
            run_eval(
                base_config,
                variant="v99",
                adapter=Path("definitely/does/not/exist"),
                test_category="non_live",
            )
        assert calls == []  # subprocess must not be invoked

    def test_subprocess_failure_surfaces(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="bfcl crash")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            run_eval(base_config, variant="v1", adapter=None, test_category="non_live")

    def test_end_to_end_writes_canonical_scores_json(self, base_config, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake_raw = {
            "overall_accuracy_non_live": 0.55,
            "overall_accuracy_live": 0.45,
            "total_irrelevance": 0.30,
            "overall_accuracy_multi_turn": 0.10,
            "overall_accuracy_agentic": 0.05,
        }

        import src.evaluation.bfcl as bfcl_mod

        monkeypatch.setattr(bfcl_mod, "parse_bfcl_score_output", lambda *a, **kw: fake_raw)

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        result_path = run_eval(base_config, variant="v1", adapter=None, test_category="all")
        assert result_path.exists()
        assert result_path.name == "scores.json"

        data = json.loads(result_path.read_text())
        assert data["variant"] == "v1"
        assert data["top_level"]["non_live"] == 0.55
        assert data["top_level"]["hallucination"] == 0.30
        assert isinstance(data["overall"], float)
        # Weights travel with results for reproducibility
        assert data["weights"]["agentic"] == 0.4
        # Generation config is captured from the config
        assert data["generation_config"]["max_tokens"] == 512


class TestParseBfclScoreOutput:
    def test_parses_real_headers(self, tmp_path):
        # Values correspond to the exact column order of REAL_OVERALL_HEADER.
        # Non-Live AST Acc = 0.55, Live Acc = 0.40, Multi Turn Acc = 0.12,
        # Web Search Acc = 0.08, Memory Acc = 0.04, Irrelevance Detection = 0.30.
        row_values = [
            "1", "45.00%", "Test Model", "http://x", "0.00",
            "1.0", "0.1", "1.5",
            "55.00%", "50.00%", "60.00%", "55.00%", "55.00%",      # non-live AST + subs
            "40.00%", "38.00%", "42.00%", "40.00%", "40.00%",      # live block
            "12.00%", "10.00%", "12.00%", "13.00%", "13.00%",      # multi-turn block
            "8.00%", "8.00%", "8.00%",                              # web search
            "4.00%", "4.00%", "4.00%", "4.00%",                     # memory
            "95.00%", "30.00%",                                     # relevance, irrelevance
            "0.03", "0.02",                                         # format sensitivity
            "TestOrg", "apache-2.0",
        ]
        assert len(row_values) == 36  # matches REAL_OVERALL_HEADER

        overall_path = tmp_path / "data_overall.csv"
        overall_path.write_text(REAL_OVERALL_HEADER + "\n" + ",".join(row_values) + "\n")

        non_live_path = tmp_path / "data_non_live.csv"
        non_live_path.write_text(
            REAL_NON_LIVE_HEADER + "\n"
            "1,Test Model,52.50%,55.00%,50.00%,48.00%,N/A,N/A,60.00%,55.00%,55.00%,50.00%\n"
        )

        raw = parse_bfcl_score_output(tmp_path)
        # non_live comes from data_non_live.csv (Non-Live Overall Acc, not Non-Live AST Acc)
        assert abs(raw["overall_accuracy_non_live"] - 0.525) < 1e-6
        assert abs(raw["overall_accuracy_live"] - 0.40) < 1e-6
        assert abs(raw["overall_accuracy_multi_turn"] - 0.12) < 1e-6
        # agentic = avg(web_search 0.08, memory 0.04) = 0.06
        assert abs(raw["overall_accuracy_agentic"] - 0.06) < 1e-6
        assert abs(raw["total_irrelevance"] - 0.30) < 1e-6
        assert abs(raw["format_sensitivity"]["accuracy_max_delta"] - 0.03) < 1e-6

    def test_missing_score_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_bfcl_score_output(tmp_path / "does-not-exist")

    def test_na_values_become_none(self, tmp_path):
        row = ["1", "N/A", "Model", "http://x", "0.00", "N/A", "N/A", "N/A"] \
              + ["N/A"] * 28
        assert len(row) == 36
        (tmp_path / "data_overall.csv").write_text(
            REAL_OVERALL_HEADER + "\n" + ",".join(row) + "\n"
        )
        raw = parse_bfcl_score_output(tmp_path)
        assert raw["overall_accuracy_live"] is None
        assert raw["overall_accuracy_agentic"] is None


class TestLoadConfig:
    def test_loads_bfcl_config(self):
        # Resolved relative to repo root; tests are expected to be run from there.
        config = load_config("configs/bfcl.yaml")
        assert config["model"]["name"] == "Qwen/Qwen2.5-3B-Instruct"
        assert config["model"]["bfcl_model_id"] == "Qwen/Qwen2.5-3B-Instruct-FC"
        assert config["generation"]["temperature"] == 0.0
        assert config["generation"]["max_tokens"] == 512
        assert config["generation"]["backend"] == "vllm"
        assert abs(sum(config["weights"].values()) - 1.0) < 1e-9

    def test_nested_override(self):
        config = load_config(
            "configs/bfcl.yaml",
            overrides={"generation": {"temperature": 0.5}},
        )
        assert config["generation"]["temperature"] == 0.5
        # Other generation keys are preserved
        assert config["generation"]["max_tokens"] == 512


class TestMergeWithExistingScores:
    """Per-bucket evaluations across separate Colab sessions must accumulate
    into one scores.json. Previously-evaluated buckets cannot be silently
    dropped when a partial follow-up run masks them as null."""

    @pytest.fixture
    def weights(self, base_config):
        return base_config["weights"]

    def _canonical(self, top_level: dict) -> dict:
        return {
            "variant": "v2",
            "top_level": dict(top_level),
            "overall": 0.0,
            "run_metadata": {},
        }

    def test_returns_input_when_existing_missing(self, tmp_path, weights):
        canonical = self._canonical({"non_live": 0.8, "live": None, "hallucination": None, "multi_turn": None, "agentic": None})
        merged = merge_with_existing_scores(canonical, tmp_path / "missing.json", weights)
        assert merged is canonical

    def test_preserves_previous_buckets(self, tmp_path, weights):
        existing_path = tmp_path / "scores.json"
        existing_path.write_text(json.dumps({
            "top_level": {"non_live": 0.79, "live": None, "hallucination": None, "multi_turn": None, "agentic": None},
        }))

        new_run = self._canonical({"non_live": None, "live": 0.65, "hallucination": None, "multi_turn": None, "agentic": None})
        merged = merge_with_existing_scores(new_run, existing_path, weights)

        assert merged["top_level"]["non_live"] == 0.79
        assert merged["top_level"]["live"] == 0.65
        assert merged["run_metadata"]["merged_with"] == ["non_live"]

    def test_recomputes_overall_after_merge(self, tmp_path, weights):
        existing_path = tmp_path / "scores.json"
        existing_path.write_text(json.dumps({
            "top_level": {"non_live": 0.8, "live": None, "hallucination": None, "multi_turn": None, "agentic": None},
        }))

        new_run = self._canonical({"non_live": None, "live": 0.6, "hallucination": None, "multi_turn": None, "agentic": None})
        merged = merge_with_existing_scores(new_run, existing_path, weights)

        # `aggregate_weighted_score` treats None as 0 and does not renormalize.
        # 0.8*0.1 (non_live) + 0.6*0.1 (live) = 0.14
        assert abs(merged["overall"] - 0.14) < 1e-9

    def test_does_not_overwrite_newer_run(self, tmp_path, weights):
        existing_path = tmp_path / "scores.json"
        existing_path.write_text(json.dumps({
            "top_level": {"non_live": 0.5, "live": None, "hallucination": None, "multi_turn": None, "agentic": None},
        }))

        # New run re-evaluated non_live and got a different score; new wins.
        new_run = self._canonical({"non_live": 0.9, "live": None, "hallucination": None, "multi_turn": None, "agentic": None})
        merged = merge_with_existing_scores(new_run, existing_path, weights)

        assert merged["top_level"]["non_live"] == 0.9

    def test_no_op_when_nothing_to_preserve(self, tmp_path, weights):
        existing_path = tmp_path / "scores.json"
        existing_path.write_text(json.dumps({
            "top_level": {"non_live": None, "live": None, "hallucination": None, "multi_turn": None, "agentic": None},
        }))

        new_run = self._canonical({"non_live": 0.5, "live": None, "hallucination": None, "multi_turn": None, "agentic": None})
        merged = merge_with_existing_scores(new_run, existing_path, weights)

        # No buckets to preserve, identity-return.
        assert merged is new_run

    def test_returns_input_when_existing_is_corrupt(self, tmp_path, weights):
        # Colab session crashes mid-write to scores.json leave truncated JSON.
        # The next run must treat that the same as a missing file, not crash.
        existing_path = tmp_path / "scores.json"
        existing_path.write_text("{truncated invalid json")
        canonical = self._canonical({"non_live": 0.8, "live": None, "hallucination": None, "multi_turn": None, "agentic": None})
        merged = merge_with_existing_scores(canonical, existing_path, weights)
        assert merged is canonical

    def test_does_not_mutate_input_run_metadata(self, tmp_path, weights):
        # A future caller passing in a re-used canonical must not see their
        # run_metadata silently mutated by the merge.
        existing_path = tmp_path / "scores.json"
        existing_path.write_text(json.dumps({
            "top_level": {"non_live": 0.79, "live": None, "hallucination": None, "multi_turn": None, "agentic": None},
        }))

        original_metadata = {"timestamp": "2026-04-27T00:00:00Z"}
        new_run = self._canonical({"non_live": None, "live": 0.65, "hallucination": None, "multi_turn": None, "agentic": None})
        new_run["run_metadata"] = original_metadata

        merged = merge_with_existing_scores(new_run, existing_path, weights)

        assert "merged_with" in merged["run_metadata"]
        assert "merged_with" not in original_metadata


class TestCoerce:
    """The bfcl harness reads CSVs that mix '%'-suffixed percentages with
    plain floats. The earlier `v / 100 if v > 1` heuristic mis-handled
    sub-1% percents like '0.62%' (real V4 multi_turn score), returning
    0.62 instead of 0.0062 and inflating the V4-weighted overall by ~18pp
    on multi_turn alone. _coerce must detect '%' explicitly."""

    def test_sub_one_percent_is_correctly_scaled(self):
        # '0.62%' is BFCL's actual V4 multi_turn output for Qwen-3B.
        assert abs(_coerce("0.62%") - 0.0062) < 1e-9

    def test_above_one_percent_is_correctly_scaled(self):
        assert abs(_coerce("23.14%") - 0.2314) < 1e-9
        assert abs(_coerce("80.08%") - 0.8008) < 1e-9

    def test_one_hundred_percent(self):
        assert _coerce("100.00%") == 1.0

    def test_plain_float_without_percent_passes_through(self):
        # `total_irrelevance` fallback from per-bucket JSON returns a fraction.
        assert _coerce("0.6") == 0.6
        assert _coerce(0.6167) == 0.6167

    def test_na_returns_none(self):
        assert _coerce("N/A") is None
        assert _coerce("") is None
        assert _coerce(None) is None

    def test_whitespace_tolerated(self):
        assert abs(_coerce(" 0.62% ") - 0.0062) < 1e-9
        assert abs(_coerce("0.62 %") - 0.0062) < 1e-9


class TestShimBfclOutputBucketMasking:
    """When --test-category is partial (e.g. non_live) but bfcl's aggregate
    CSVs reflect prior-run data for other buckets, the shim must trust those
    real values instead of nulling them out as 'didn't run'. Otherwise
    merge_with_existing_scores preserves a stale per-bucket-JSON value (the
    V3 hallucination 0.6 vs CSV 0.6247 bug)."""

    def test_partial_run_keeps_real_csv_values_from_prior_runs(self, base_config):
        # Simulate a non_live re-run where the CSV has prior live, multi_turn,
        # and hallucination data populated by earlier per-bucket evals.
        raw = {
            "overall_accuracy_non_live": 0.80,
            "overall_accuracy_live": 0.6765,        # set by prior live run
            "total_irrelevance": 0.6247,            # set by prior irrelevance run
            "overall_accuracy_multi_turn": 0.0062,  # set by prior multi_turn run
            "overall_accuracy_agentic": None,       # genuinely never run
        }
        canonical = shim_bfcl_output(
            raw, base_config, variant="v3", adapter=None, test_category="non_live"
        )
        assert canonical["top_level"]["non_live"] == 0.80
        assert canonical["top_level"]["live"] == 0.6765
        assert canonical["top_level"]["hallucination"] == 0.6247
        assert canonical["top_level"]["multi_turn"] == 0.0062
        assert canonical["top_level"]["agentic"] is None

    def test_partial_run_still_masks_zero_placeholders(self, base_config):
        # The original masking concern: bfcl writes Live Acc=0% when no live
        # results exist on disk. Those zeros are placeholders, not honest
        # scores, and must still be masked.
        raw = {
            "overall_accuracy_non_live": 0.80,
            "overall_accuracy_live": 0.0,
            "overall_accuracy_multi_turn": 0.0,
            "total_irrelevance": None,
            "overall_accuracy_agentic": None,
        }
        canonical = shim_bfcl_output(
            raw, base_config, variant="v1", adapter=None, test_category="non_live"
        )
        assert canonical["top_level"]["non_live"] == 0.80
        assert canonical["top_level"]["live"] is None
        assert canonical["top_level"]["multi_turn"] is None
        assert canonical["top_level"]["hallucination"] is None
        assert canonical["top_level"]["agentic"] is None


class TestCommittedScoresConsistency:
    """Guard against the failure mode where a stored `overall` doesn't match
    `aggregate_weighted_score(top_level, weights)`. PR #25 originally landed
    V1 with a renormalized overall (0.7967) that diverged from the rest of
    the codebase. Fixed in PR #35; this test prevents recurrence.
    """

    def test_all_committed_scores_match_aggregator(self):
        results_root = Path("results")
        scores_paths = sorted(results_root.glob("*/scores.json"))
        if not scores_paths:
            pytest.skip("No committed score files found")
        for scores_path in scores_paths:
            scores = json.loads(scores_path.read_text())
            recomputed = aggregate_weighted_score(scores["top_level"], scores["weights"])
            assert abs(scores["overall"] - recomputed) < 1e-9, (
                f"{scores_path} overall ({scores['overall']}) "
                f"diverges from aggregate_weighted_score result ({recomputed}). "
                "Either the stored overall is stale or aggregate_weighted_score's "
                "semantics changed. Don't paper over: figure out which."
            )
