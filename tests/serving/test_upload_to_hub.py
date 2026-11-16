import json
import re

import pytest

from src.serving.upload_to_hub import build_model_card, main


SAMPLE_SCORES = {
    "variant": "v3-sft-masked",
    "model": "Qwen/Qwen2.5-3B-Instruct",
    "adapter": "arnursb/info290-v3-sft-masked",
    "bfcl_version": "2026.3.23",
    "top_level": {
        "non_live": 0.8234,
        "live": 0.7512,
        "hallucination": 0.6812,
        "multi_turn": None,
        "agentic": None,
    },
    "weights": {
        "agentic": 0.4,
        "multi_turn": 0.3,
        "live": 0.1,
        "non_live": 0.1,
        "hallucination": 0.1,
    },
    "overall": 0.7625,
}


def test_card_has_yaml_frontmatter():
    card = build_model_card(SAMPLE_SCORES, "SFT + masking", "Qwen/Qwen2.5-3B-Instruct", "arnursb/info290-fc-slm")
    assert card.startswith("---\n")
    end = card.index("\n---\n", 4)
    fm = card[4:end]
    assert "license: apache-2.0" in fm
    assert "base_model: Qwen/Qwen2.5-3B-Instruct" in fm
    assert "library_name: peft" in fm
    assert "tags:" in fm
    assert "- function-calling" in fm
    assert "- qwen2.5" in fm


def test_card_contains_required_sections():
    card = build_model_card(SAMPLE_SCORES, "SFT + masking", "Qwen/Qwen2.5-3B-Instruct", "arnursb/info290-fc-slm")
    for section in (
        "## Description",
        "## Intended use",
        "## Training data",
        "## Training procedure",
        "## Evaluation results",
        "## Usage",
        "## Limitations",
        "## Citation",
    ):
        assert section in card, f"missing section: {section}"


def test_card_formats_scores_as_percentages():
    card = build_model_card(SAMPLE_SCORES, "SFT + masking", "Qwen/Qwen2.5-3B-Instruct", "arnursb/info290-fc-slm")
    assert "82.34%" in card
    assert "75.12%" in card
    assert "68.12%" in card
    assert "76.25%" in card


def test_card_renders_null_buckets_as_na():
    card = build_model_card(SAMPLE_SCORES, "SFT + masking", "Qwen/Qwen2.5-3B-Instruct", "arnursb/info290-fc-slm")
    multi_turn_row = next(line for line in card.splitlines() if line.startswith("| Multi-Turn"))
    assert "N/A" in multi_turn_row


def test_card_usage_uses_peft_pattern():
    target = "arnursb/info290-fc-slm"
    card = build_model_card(SAMPLE_SCORES, "SFT + masking", "Qwen/Qwen2.5-3B-Instruct", target)
    assert "PeftModel.from_pretrained" in card
    assert target in card


def test_card_without_scores_still_well_formed():
    card = build_model_card(None, "GRPO + masking", "Qwen/Qwen2.5-3B-Instruct", "arnursb/info290-fc-slm")
    assert card.startswith("---\n")
    assert "## Evaluation results" in card
    assert "not embedded" in card


def test_card_title_uses_variant_name():
    card = build_model_card(SAMPLE_SCORES, "GRPO + curriculum", "Qwen/Qwen2.5-3B-Instruct", "arnursb/x")
    assert "# INFO 290 Function-Calling SLM (GRPO + curriculum)" in card


def test_main_requires_source_and_target(capsys):
    with pytest.raises(SystemExit):
        main([])
    err = capsys.readouterr().err
    assert "required" in err or "arguments" in err


def test_main_fails_without_hf_token(monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = main(["--source-repo", "arnursb/info290-v3-sft-masked", "--target-repo", "arnursb/info290-fc-slm"])
    assert rc == 2
    assert "HF_TOKEN" in capsys.readouterr().err


def test_card_omits_masking_text_when_unmasked():
    # V2 SFT (unmasked) and V4 GRPO (cold-start, unmasked) must not advertise
    # function masking on the public hub.
    card = build_model_card(SAMPLE_SCORES, "GRPO cold-start", "Qwen/Qwen2.5-3B-Instruct", "arnursb/x", masked=False)
    assert "Hammer-style function masking" not in card
    assert "67% of examples" not in card


def test_card_includes_masking_text_when_masked():
    card = build_model_card(SAMPLE_SCORES, "SFT + masking", "Qwen/Qwen2.5-3B-Instruct", "arnursb/x", masked=True)
    assert "Hammer-style function masking" in card
    assert "67% of examples" in card


def test_main_handles_missing_scores_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HF_TOKEN", "fake")
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--source-repo", "arnursb/x",
            "--target-repo", "arnursb/y",
            "--scores-json", str(tmp_path / "missing.json"),
            "--dry-run",
        ])
    assert exc_info.value.code == 1
    assert "scores file not found" in capsys.readouterr().err


def test_main_handles_malformed_scores_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HF_TOKEN", "fake")
    bad = tmp_path / "scores.json"
    bad.write_text("{truncated invalid json")
    with pytest.raises(SystemExit) as exc_info:
        main([
            "--source-repo", "arnursb/x",
            "--target-repo", "arnursb/y",
            "--scores-json", str(bad),
            "--dry-run",
        ])
    assert exc_info.value.code == 1
    assert "malformed JSON" in capsys.readouterr().err


def test_main_dry_run_skips_token_check(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(SAMPLE_SCORES))
    rc = main([
        "--source-repo", "arnursb/info290-v3-sft-masked",
        "--target-repo", "arnursb/info290-fc-slm",
        "--scores-json", str(scores_path),
        "--variant-name", "SFT + masking",
        "--dry-run",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert "arnursb/info290-fc-slm" in out
    assert re.search(r"82\.34%", out)
