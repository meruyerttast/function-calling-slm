"""Download and explore the xLAM-function-calling-60k dataset.

Downloads from HuggingFace, prints distribution stats, and saves raw data locally.

Usage:
    python -m src.data.download --output data/raw/xlam_60k.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


def download_xlam(output_path: str) -> list[dict]:
    """Download xLAM-function-calling-60k and save as JSON."""
    print("Downloading Salesforce/xlam-function-calling-60k ...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    print(f"Downloaded {len(ds)} examples")

    # The dataset has string fields that contain JSON.
    # Parse them into proper dicts.
    examples = []
    for row in ds:
        example = {
            "query": row["query"],
            "tools": json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"],
            "answers": json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"],
        }
        examples.append(example)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Saved to {output_path}")
    return examples


def print_stats(examples: list[dict]) -> None:
    """Print dataset distribution stats."""
    num_tools_dist = Counter()
    num_calls_dist = Counter()
    call_types = Counter()

    for ex in examples:
        tools = ex["tools"]
        answers = ex["answers"]

        num_tools_dist[len(tools)] += 1
        num_calls_dist[len(answers)] += 1

        if len(answers) == 0:
            call_types["no_call (irrelevance)"] += 1
        elif len(answers) == 1:
            call_types["single_call"] += 1
        else:
            call_types["parallel_calls"] += 1

    print(f"\nTotal examples: {len(examples)}")

    print("\n-- Tools per example --")
    for k in sorted(num_tools_dist.keys()):
        print(f"  {k} tools: {num_tools_dist[k]}")

    print("\n-- Function calls per example --")
    for k in sorted(num_calls_dist.keys()):
        print(f"  {k} calls: {num_calls_dist[k]}")

    print("\n-- Call type breakdown --")
    for k, v in call_types.most_common():
        pct = 100 * v / len(examples)
        print(f"  {k}: {v} ({pct:.1f}%)")

    # Sample a few parameter counts
    param_counts = []
    for ex in examples:
        for ans in ex["answers"]:
            param_counts.append(len(ans.get("arguments", {})))
    if param_counts:
        avg_params = sum(param_counts) / len(param_counts)
        print(f"\nAvg parameters per call: {avg_params:.1f}")
        print(f"Max parameters in a single call: {max(param_counts)}")


def main():
    parser = argparse.ArgumentParser(description="Download xLAM-60k dataset")
    parser.add_argument(
        "--output",
        default="data/raw/xlam_60k.json",
        help="Output path for raw dataset JSON",
    )
    args = parser.parse_args()

    examples = download_xlam(args.output)
    print_stats(examples)


if __name__ == "__main__":
    main()
