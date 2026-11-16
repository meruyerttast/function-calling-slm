"""Promote a private staging adapter repo to a public release repo with a model card.

After full BFCL V4 evaluation, the team selects the best variant and runs:
    python -m src.serving.upload_to_hub \
        --source-repo arnursb/info290-v3-sft-masked \
        --target-repo arnursb/info290-function-calling-slm-3b \
        --scores-json results/v3-sft-masked/scores.json \
        --variant-name "SFT + function masking"
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LICENSE = "apache-2.0"
TAGS = ["function-calling", "tool-use", "qwen2.5", "peft", "lora"]
DATASET = "Salesforce/xlam-function-calling-60k"

BUCKET_LABELS = {
    "non_live": "Non-Live",
    "live": "Live",
    "multi_turn": "Multi-Turn",
    "agentic": "Agentic",
    "hallucination": "Hallucination",
}


def _format_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def _scores_table(scores: dict) -> str:
    top = scores.get("top_level", {})
    weights = scores.get("weights", {})
    rows = ["| Bucket | Weight | Score |", "|---|---|---|"]
    for key, label in BUCKET_LABELS.items():
        weight = weights.get(key)
        weight_str = f"{int(weight * 100)}%" if weight is not None else "-"
        rows.append(f"| {label} | {weight_str} | {_format_score(top.get(key))} |")
    overall = scores.get("overall")
    rows.append(f"| **Overall (V4-weighted)** | 100% | **{_format_score(overall)}** |")
    return "\n".join(rows)


def build_model_card(
    scores: dict | None,
    variant_name: str,
    base_model: str,
    target_repo: str,
    *,
    masked: bool = True,
) -> str:
    title = f"INFO 290 Function-Calling SLM ({variant_name})" if variant_name else "INFO 290 Function-Calling SLM"
    tags_yaml = "\n".join(f"- {t}" for t in TAGS)
    frontmatter = (
        "---\n"
        f"license: {LICENSE}\n"
        f"base_model: {base_model}\n"
        "library_name: peft\n"
        f"datasets:\n- {DATASET}\n"
        f"tags:\n{tags_yaml}\n"
        "---\n"
    )

    if scores is not None:
        bfcl_version = scores.get("bfcl_version", "unknown")
        eval_section = (
            f"Evaluated on the Berkeley Function Calling Leaderboard V4 ({bfcl_version}). "
            "Buckets and weights follow the official V4 schema.\n\n"
            f"{_scores_table(scores)}\n"
        )
    else:
        eval_section = (
            "BFCL V4 scores were not embedded at upload time. See the project repo for the latest numbers.\n"
        )

    masking_clause = (
        "with Hammer-style function masking (67% of examples) to encourage reasoning from "
        "natural-language tool descriptions rather than name memorization "
        if masked else ""
    )
    masking_detail = (
        "Function masking is applied to 67% of examples (function names, parameter names, "
        "and default values are perturbed) to force generalization to unseen APIs.\n\n"
        if masked else ""
    )

    return frontmatter + (
        f"\n# {title}\n\n"
        "## Description\n\n"
        "LoRA adapter (rank 64) for Qwen2.5-3B-Instruct, fine-tuned for function calling at the sub-3B "
        "parameter scale. Trained on a 4,000-example curated subset of Salesforce's xLAM-function-calling-60k "
        f"{masking_clause}"
        ". Final coursework deliverable for UC Berkeley "
        "INFO 290 (Generative AI, Spring 2026).\n\n"
        "## Intended use\n\n"
        "Function calling and tool use in single-turn and parallel-call settings. Suitable for on-device "
        "and budget-constrained deployments where frontier models are too costly. Not intended for "
        "general dialogue, factual QA, or safety-critical decisions.\n\n"
        "## Training data\n\n"
        f"4,000 examples sampled from [{DATASET}](https://huggingface.co/datasets/{DATASET}) (CC-BY-4.0) "
        "via a four-stage pipeline: validation, deduplication, balanced sampling (60% single-call, "
        "30% parallel-call, 10% no-call target), and parameter-type diversity scoring. "
        f"{masking_detail}"
        "## Training procedure\n\n"
        "- QLoRA, rank 64, alpha 128, dropout 0.05, NF4 quantization\n"
        "- Adapters on all attention projection and MLP matrices\n"
        "- Optimizer: paged AdamW 8-bit, fp16 precision\n"
        "- Schedule: cosine LR with 5% warmup\n"
        "- Stack: TRL `SFTTrainer` / `GRPOTrainer` on PEFT\n"
        "- Configs: `configs/sft.yaml`, `configs/grpo.yaml` in the project repo\n\n"
        "## Evaluation results\n\n"
        f"{eval_section}\n"
        "## Usage\n\n"
        "This is a LoRA adapter; load it on top of the base model with PEFT:\n\n"
        "```python\n"
        "from peft import PeftModel\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        f'base = AutoModelForCausalLM.from_pretrained("{base_model}", device_map="auto")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{base_model}")\n'
        f'model = PeftModel.from_pretrained(base, "{target_repo}")\n'
        "```\n\n"
        "For OpenAI-compatible serving, see `src/serving/` in the project repo (vLLM scaffold with "
        "LoRA support).\n\n"
        "## Limitations\n\n"
        "- The training subset contains zero no-call (irrelevance) examples, which limits hallucination "
        "performance against BFCL V4's 10% Hallucination bucket.\n"
        "- QLoRA adapters typically lose a few points vs. full fine-tunes on multi-turn dialog.\n"
        "- Source data is predominantly Python; transfer to Java and JavaScript syntax families is weaker.\n"
        "- Trained and evaluated only in English.\n\n"
        "## Citation\n\n"
        "```bibtex\n"
        "@misc{info290-fc-slm-2026,\n"
        "  title  = {Pushing the Sub-3B Function-Calling Frontier with GRPO and Function Masking},\n"
        "  author = {INFO 290 Final Project Team},\n"
        "  year   = {2026},\n"
        "  note   = {UC Berkeley INFO 290, Spring 2026}\n"
        "}\n"
        "```\n"
    )


def _load_scores(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        print(f"ERROR: scores file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: malformed JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-repo", required=True, help="Private staging repo (arnursb/info290-vX-...)")
    parser.add_argument("--target-repo", required=True, help="Public release repo (created if missing)")
    parser.add_argument("--scores-json", help="Path to results/<variant>/scores.json to embed in the card")
    parser.add_argument("--variant-name", default="", help="Human-readable variant name for the card title")
    parser.add_argument(
        "--masked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether the variant was trained with function masking (controls model card text). "
             "Use --no-masked for V2 SFT or V4 GRPO (cold-start, unmasked).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan without uploading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("ERROR: HF_TOKEN env var is required.", file=sys.stderr)
        return 2

    scores = _load_scores(args.scores_json)
    card = build_model_card(scores, args.variant_name, BASE_MODEL, args.target_repo, masked=args.masked)

    if args.dry_run:
        print(f"[dry-run] would snapshot {args.source_repo} and push to {args.target_repo} (public)")
        print(f"[dry-run] model card preview ({len(card)} chars):\n")
        print(card)
        return 0

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmp:
        local_dir = snapshot_download(
            repo_id=args.source_repo,
            repo_type="model",
            local_dir=tmp,
            token=token,
        )
        Path(local_dir, "README.md").write_text(card)

        api.create_repo(args.target_repo, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            folder_path=local_dir,
            repo_id=args.target_repo,
            repo_type="model",
            commit_message=f"promote from {args.source_repo}",
        )

        listed = set(api.list_repo_files(args.target_repo))
        required = {"adapter_config.json", "README.md"}
        missing = required - listed
        if missing:
            print(f"ERROR: upload verification failed; missing {sorted(missing)} on hub.", file=sys.stderr)
            return 1

    url = f"https://huggingface.co/{args.target_repo}"
    print(f"Promoted {args.source_repo} -> {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
