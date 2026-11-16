"""BFCL V4 evaluation harness (Issue #5).

Orchestrates the official bfcl-eval CLI to score Qwen2.5-3B function-calling
variants against the Berkeley Function Calling Leaderboard V4. Two subprocess
invocations per run (bfcl generate then bfcl evaluate), followed by a shim
that converts the upstream score output into our canonical scores.json shape.

Usage:
    python -m src.evaluation.bfcl --variant v1
    python -m src.evaluation.bfcl --variant v2 --adapter checkpoints/sft/v2/final
    python -m src.evaluation.bfcl --variant v1 --test-category non_live --dry-run
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.training.rewards import parse_tool_calls


RAW_TO_CANONICAL_BUCKET = {
    "overall_accuracy_non_live": "non_live",
    "overall_accuracy_live": "live",
    "total_irrelevance": "hallucination",
    "overall_accuracy_multi_turn": "multi_turn",
    "overall_accuracy_agentic": "agentic",
}

CANONICAL_BUCKETS = ("non_live", "live", "hallucination", "multi_turn", "agentic")

# Maps bfcl --test-category values to the top-level V4 buckets they populate.
# Used to null out buckets that weren't evaluated in a partial run. Top-level
# names map to themselves; sub-category names roll up to their parent bucket.
# "all" means every bucket is in play; anything unrecognized also falls through
# to "all" so we don't silently drop real scores.
_CATEGORY_TO_BUCKETS: dict[str, frozenset[str]] = {
    "all": frozenset(CANONICAL_BUCKETS),
    "non_live": frozenset({"non_live"}),
    "live": frozenset({"live"}),
    "hallucination": frozenset({"hallucination"}),
    "multi_turn": frozenset({"multi_turn"}),
    "agentic": frozenset({"agentic"}),
    "simple_python": frozenset({"non_live"}),
    "simple_java": frozenset({"non_live"}),
    "simple_javascript": frozenset({"non_live"}),
    "multiple": frozenset({"non_live"}),
    "parallel": frozenset({"non_live"}),
    "parallel_multiple": frozenset({"non_live"}),
    "irrelevance": frozenset({"hallucination"}),
    "live_simple": frozenset({"live"}),
    "live_multiple": frozenset({"live"}),
    "live_parallel": frozenset({"live"}),
    "live_parallel_multiple": frozenset({"live"}),
    "live_irrelevance": frozenset({"live"}),
}


def _buckets_evaluated(test_category: str) -> frozenset[str]:
    return _CATEGORY_TO_BUCKETS.get(test_category, frozenset(CANONICAL_BUCKETS))


def load_config(config_path: str, overrides: dict | None = None) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def aggregate_weighted_score(scores: dict, weights: dict) -> float:
    total = 0.0
    for bucket, weight in weights.items():
        value = scores.get(bucket)
        if value is None:
            continue
        total += float(value) * float(weight)
    return total


def build_generate_cmd(
    config: dict,
    variant: str,
    adapter: Path | str | None = None,
    test_category: str = "all",
) -> list[str]:
    gen_cfg = config["generation"]
    bfcl_cfg = config["bfcl"]
    bfcl_model_id = config["model"]["bfcl_model_id"]
    # bfcl-eval resolves --result-dir relative to its own install dir, not CWD,
    # so pass an absolute path to keep outputs in the project's results/ tree.
    result_dir = str(Path(bfcl_cfg["result_dir_template"].format(variant=variant)).resolve())

    cmd = [
        "bfcl", "generate",
        "--model", bfcl_model_id,
        "--test-category", test_category,
        "--temperature", str(gen_cfg["temperature"]),
        "--backend", gen_cfg["backend"],
        "--result-dir", result_dir,
        "--allow-overwrite",
    ]

    if adapter is not None:
        adapter_path = Path(adapter).resolve()
        cmd.extend([
            "--enable-lora",
            "--max-lora-rank", str(config["lora"]["max_rank"]),
            "--lora-modules", f'{variant}={adapter_path}',
        ])

    return cmd


def build_evaluate_cmd(config: dict, variant: str, test_category: str = "all") -> list[str]:
    bfcl_cfg = config["bfcl"]
    bfcl_model_id = config["model"]["bfcl_model_id"]
    # Same rationale as build_generate_cmd: upstream treats these as relative to
    # bfcl-eval's install dir unless absolute.
    result_dir = str(Path(bfcl_cfg["result_dir_template"].format(variant=variant)).resolve())
    score_dir = str(Path(bfcl_cfg["score_dir_template"].format(variant=variant)).resolve())

    return [
        "bfcl", "evaluate",
        "--model", bfcl_model_id,
        "--test-category", test_category,
        "--result-dir", result_dir,
        "--score-dir", score_dir,
        "--partial-eval",
    ]


def merge_with_existing_scores(canonical: dict, existing_path: Path, weights: dict) -> dict:
    """Preserve previously-evaluated buckets when a partial run would overwrite them.

    Per-bucket evals run across separate Colab sessions need to accumulate into
    one scores.json. After shim_bfcl_output masks unevaluated buckets to None,
    re-fill them from the existing file so a `live` run after a `non_live` run
    does not silently drop the non_live result.

    Sets `run_metadata.merged_with` to the list of buckets that were preserved
    from the prior scores.json (i.e. buckets NOT computed in this partial run).
    The current bucket is intentionally absent because it was freshly evaluated;
    to enumerate ALL buckets present in scores.json, read `top_level` keys
    where the value is non-None.
    """
    if not existing_path.exists():
        return canonical

    # A Colab session crash mid-write leaves a truncated scores.json. Treat it
    # the same as missing rather than turning a recoverable disconnect into an
    # unrecoverable one - which is precisely the failure mode this merge guards.
    try:
        with open(existing_path) as f:
            existing = json.load(f)
    except json.JSONDecodeError:
        return canonical

    existing_top = existing.get("top_level") or {}
    merged = dict(canonical["top_level"])
    preserved = []
    for bucket, score in merged.items():
        if score is None and existing_top.get(bucket) is not None:
            merged[bucket] = existing_top[bucket]
            preserved.append(bucket)

    if not preserved:
        return canonical

    canonical = dict(canonical)
    canonical["top_level"] = merged
    canonical["overall"] = aggregate_weighted_score(merged, weights)
    canonical["run_metadata"] = dict(canonical.get("run_metadata") or {})
    canonical["run_metadata"]["merged_with"] = preserved
    return canonical


def shim_bfcl_output(
    raw: dict,
    config: dict,
    variant: str,
    adapter: Path | str | None,
    test_category: str = "all",
    adapter_source: str | None = None,
) -> dict:
    # bfcl's data_overall.csv reports 0% for buckets that weren't evaluated in
    # a partial run (e.g. Live Acc = 0% when --test-category non_live). Mask
    # those out so downstream consumers see null instead of a misleading zero.
    #
    # But: when prior partial runs left score files for other buckets on disk,
    # bfcl's aggregator populates those columns with real (non-zero) values.
    # Trust the CSV for non-zero values regardless of this run's test_category;
    # mask only when the bucket wasn't evaluated AND the CSV value is 0.0
    # (the "didn't run" placeholder). Without this, a partial non_live re-run
    # silently nulls out a real Irrelevance Detection or Multi Turn Acc that
    # the CSV correctly reports from prior runs, which then triggers
    # merge_with_existing_scores to preserve a stale value (the V3 hallucination
    # 0.6 vs CSV 0.6247 bug).
    evaluated = _buckets_evaluated(test_category)
    top_level = {bucket: None for bucket in CANONICAL_BUCKETS}
    for raw_key, canonical_key in RAW_TO_CANONICAL_BUCKET.items():
        raw_val = raw.get(raw_key)
        if raw_val is None:
            continue
        raw_val = float(raw_val)
        if canonical_key not in evaluated and raw_val == 0.0:
            continue
        top_level[canonical_key] = raw_val

    weights = config["weights"]
    overall = aggregate_weighted_score(top_level, weights)

    fs_meta = raw.get("format_sensitivity") or {}
    side_metrics = {
        "format_sensitivity_max_delta": fs_meta.get("accuracy_max_delta"),
        "format_sensitivity_std": fs_meta.get("accuracy_std"),
    }

    # Prefer the portable identifier (typically a HF Hub repo id like
    # "user/info290-v2-sft") when the caller provides it; fall back to the
    # local path that vLLM was pointed at. The local path lands as a
    # downstream-unfriendly absolute string when the eval runs on HF Jobs
    # and the adapter is fetched via snapshot_download into a runner-only
    # cache directory, which is unreproducible elsewhere.
    if adapter_source is not None:
        adapter_field: str | None = adapter_source
    elif adapter is not None:
        adapter_field = str(adapter)
    else:
        adapter_field = None

    return {
        "variant": variant,
        "model": config["model"]["name"],
        "adapter": adapter_field,
        "bfcl_version": config["bfcl"]["version_pin"],
        "generation_config": {
            "temperature": config["generation"]["temperature"],
            "top_p": config["generation"]["top_p"],
            "max_tokens": config["generation"]["max_tokens"],
        },
        "top_level": top_level,
        "weights": weights,
        "overall": overall,
        "side_metrics": side_metrics,
        "run_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }


def parse_bfcl_score_output(score_dir: Path, bfcl_model_id: str | None = None) -> dict:
    """Read bfcl's score CSVs and extract V4 top-level bucket scores.

    Column names verified against upstream `constants/column_headers.py`
    (COLUMNS_OVERALL) and the row assembly at `eval_runner_helper.py:521-560`.

    Upstream writes 4-5 CSVs per run. The top-level buckets used in the V4
    weighted overall map to columns in these files:
      - non_live  = data_non_live.csv  :: "Non-Live Overall Acc"
      - live      = data_overall.csv   :: "Live Acc"
      - multi_turn= data_overall.csv   :: "Multi Turn Acc"
      - agentic   = avg(data_overall.csv :: "Web Search Acc",
                        data_overall.csv :: "Memory Acc")
      - hallucination (total_irrelevance) = data_overall.csv :: "Irrelevance Detection"

    `data_overall.csv` has no single Agentic column; upstream computes it as
    `unweighted_avg(summary_web_search, summary_memory)` at
    `eval_runner_helper.py:455`. We reproduce that averaging here.

    When `bfcl evaluate --test-category irrelevance` runs alone, the
    Irrelevance Detection column in data_overall.csv is N/A (the column is
    only populated once non_live aggregation runs). The per-bucket JSON at
    <score_dir>/<bfcl_model_id>/non_live/BFCL_v4_irrelevance_score.json holds
    the value as its first-line header. Pass bfcl_model_id to enable that
    fallback.
    """
    import csv

    overall_csv = score_dir / "data_overall.csv"
    if not overall_csv.exists():
        raise FileNotFoundError(
            f"Expected bfcl score output at {overall_csv}. "
            "Check that `bfcl evaluate` ran and --score-dir is correct."
        )

    with open(overall_csv, newline="") as f:
        overall_rows = list(csv.DictReader(f))
    if not overall_rows:
        raise ValueError(f"{overall_csv} contains no rows")
    overall_row = overall_rows[0]

    non_live_overall = None
    non_live_csv = score_dir / "data_non_live.csv"
    if non_live_csv.exists():
        with open(non_live_csv, newline="") as f:
            nl_rows = list(csv.DictReader(f))
        if nl_rows:
            non_live_overall = _coerce(nl_rows[0].get("Non-Live Overall Acc"))

    web_search = _coerce(overall_row.get("Web Search Acc"))
    memory = _coerce(overall_row.get("Memory Acc"))
    agentic = None
    if web_search is not None and memory is not None:
        agentic = (web_search + memory) / 2.0
    elif web_search is not None:
        agentic = web_search
    elif memory is not None:
        agentic = memory

    raw = {
        "overall_accuracy_non_live": non_live_overall,
        "overall_accuracy_live": _coerce(overall_row.get("Live Acc")),
        "total_irrelevance": _coerce(overall_row.get("Irrelevance Detection")),
        "overall_accuracy_multi_turn": _coerce(overall_row.get("Multi Turn Acc")),
        "overall_accuracy_agentic": agentic,
        "format_sensitivity": {
            "accuracy_max_delta": _coerce(overall_row.get("Format Sensitivity Max Delta")),
            "accuracy_std": _coerce(overall_row.get("Format Sensitivity Standard Deviation")),
        },
    }

    if raw["total_irrelevance"] is None and bfcl_model_id is not None:
        per_bucket = (
            score_dir
            / bfcl_model_id.replace("/", "_")
            / "non_live"
            / "BFCL_v4_irrelevance_score.json"
        )
        if per_bucket.exists():
            try:
                with open(per_bucket) as f:
                    header = json.loads(f.readline())
                if isinstance(header, dict) and header.get("accuracy") is not None:
                    raw["total_irrelevance"] = float(header["accuracy"])
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    # Debug fallback: if all five buckets are None, surface the headers we saw
    # so a wrong column name is a one-iteration fix instead of debug-in-the-dark.
    scoring_buckets = [
        raw["overall_accuracy_non_live"],
        raw["overall_accuracy_live"],
        raw["total_irrelevance"],
        raw["overall_accuracy_multi_turn"],
        raw["overall_accuracy_agentic"],
    ]
    if all(v is None for v in scoring_buckets):
        print(
            f"[bfcl parse] All scoring buckets are None. Headers in {overall_csv.name}:\n"
            f"  {list(overall_row.keys())}",
            file=sys.stderr,
        )

    return raw


def _coerce(value) -> float | None:
    """Convert a bfcl CSV cell to a 0..1 fraction.

    bfcl writes percentages with a trailing '%' (e.g. '23.14%', '0.62%'). The
    earlier `v / 100 if v > 1` magnitude heuristic mis-handled sub-1%
    percents: '0.62%' became 0.62 instead of 0.0062, inflating the V4-weighted
    overall by ~18pp on multi_turn alone. Detect '%' explicitly instead.
    """
    if value is None or value in ("", "N/A"):
        return None
    try:
        s = str(value).strip()
        if s.endswith("%"):
            return float(s[:-1].strip()) / 100.0
        return float(s)
    except (ValueError, TypeError):
        return None


def dump_diagnostic_generations(score_dir: Path, diagnostics_dir: Path, limit: int = 10) -> None:
    """Log the first N generations via parse_tool_calls for debugging (R9)."""
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    gen_files = sorted(score_dir.glob("*.json"))
    if not gen_files:
        return
    output = diagnostics_dir / "first_10_generations.jsonl"
    count = 0
    with open(output, "w") as out:
        for gen_file in gen_files:
            try:
                with open(gen_file) as f:
                    for line in f:
                        if count >= limit:
                            return
                        entry = json.loads(line)
                        result_text = entry.get("result") or entry.get("completion") or ""
                        parsed = parse_tool_calls(result_text) if isinstance(result_text, str) else []
                        out.write(json.dumps({
                            "file": gen_file.name,
                            "id": entry.get("id"),
                            "raw_result": result_text,
                            "parsed_tool_calls": parsed,
                        }) + "\n")
                        count += 1
            except (json.JSONDecodeError, OSError):
                continue


def run_eval(
    config: dict,
    variant: str,
    adapter: Path | str | None = None,
    test_category: str = "all",
    dry_run: bool = False,
    merge: bool = True,
    adapter_source: str | None = None,
) -> Path:
    if adapter is not None:
        adapter_path = Path(adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    output_dir = Path(config["output_dir"]) / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_cmd = build_generate_cmd(config, variant, adapter, test_category)
    evaluate_cmd = build_evaluate_cmd(config, variant, test_category)

    if dry_run:
        print("Would run:")
        print(" ", " ".join(generate_cmd))
        print(" ", " ".join(evaluate_cmd))
        return output_dir / "scores.json"

    for step, cmd in [("generate", generate_cmd), ("evaluate", evaluate_cmd)]:
        print(f"[bfcl {step}] {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr,
            )

    score_dir = Path(config["bfcl"]["score_dir_template"].format(variant=variant)).resolve()
    raw = parse_bfcl_score_output(score_dir, config["model"]["bfcl_model_id"])

    canonical = shim_bfcl_output(raw, config, variant, adapter, test_category, adapter_source)

    scores_path = output_dir / "scores.json"
    if merge:
        canonical = merge_with_existing_scores(canonical, scores_path, config["weights"])
    with open(scores_path, "w") as f:
        json.dump(canonical, f, indent=2)

    result_dir = Path(config["bfcl"]["result_dir_template"].format(variant=variant)).resolve()
    diagnostics_dir = output_dir / "diagnostics"
    dump_diagnostic_generations(result_dir, diagnostics_dir)

    print(f"\nCanonical scores written to {scores_path}")
    print(f"Overall score: {canonical['overall']:.4f}")
    for bucket, score in canonical["top_level"].items():
        display = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {bucket:15s} = {display}")

    return scores_path


def main():
    parser = argparse.ArgumentParser(description="BFCL V4 evaluation harness")
    parser.add_argument("--config", default="configs/bfcl.yaml")
    parser.add_argument("--variant", required=True, help="Variant name, e.g. v1, v2, v3")
    parser.add_argument("--adapter", default=None, help="Path to a PEFT adapter directory (optional)")
    parser.add_argument(
        "--adapter-source",
        default=None,
        help=(
            "Portable adapter identifier to record in scores.json (e.g. an HF Hub "
            "repo id like 'user/repo'). When omitted, falls back to --adapter, which "
            "may be a runner-local cache path that is not reproducible."
        ),
    )
    parser.add_argument("--model", default=None, help="Override model name from config")
    parser.add_argument(
        "--test-category",
        default="all",
        help="BFCL test category: all, agentic, non_live, live, multi_turn, or a sub-category id",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print subprocess commands without executing")
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Overwrite scores.json instead of merging with previously-evaluated buckets.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.model is not None:
        config["model"]["name"] = args.model

    run_eval(
        config,
        variant=args.variant,
        adapter=args.adapter,
        test_category=args.test_category,
        dry_run=args.dry_run,
        merge=not args.no_merge,
        adapter_source=args.adapter_source,
    )


if __name__ == "__main__":
    main()
