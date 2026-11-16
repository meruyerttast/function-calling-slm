"""Score formatted training examples by difficulty and emit a curriculum-sorted JSONL.

Heuristic difficulty score, ascending = easier first. Following ToolACE-R
(arXiv:2504.01400) but trading the model-aware scoring for a structural
heuristic that is reproducible in seconds without GPU inference.

Three signals contribute to the score:

1. Number of ground-truth calls. A parallel call is harder than a single call.
2. Number of total parameters across calls. More parameters means more places
   to err.
3. Number of available tools, weighted small. Larger tool sets force broader
   discrimination, but the differential in difficulty is modest compared to
   call complexity.

    difficulty = num_calls + num_params + 0.1 * num_tools

Usage:
    python -m src.data.curriculum \
        --input data/processed/xlam_4k_masked_formatted.jsonl \
        --output data/processed/xlam_4k_masked_curriculum_formatted.jsonl
"""

import argparse
import json
from pathlib import Path


def example_difficulty(example: dict) -> float:
    tools = example.get("tools") or []
    ground_truth = example.get("ground_truth") or []

    num_calls = len(ground_truth)
    num_params = sum(len((call or {}).get("arguments") or {}) for call in ground_truth)
    num_tools = len(tools)

    return num_calls + num_params + 0.1 * num_tools


def sort_by_curriculum(examples: list[dict]) -> list[dict]:
    return sorted(examples, key=example_difficulty)


def main():
    parser = argparse.ArgumentParser(description="Sort formatted JSONL ascending by difficulty for curriculum training.")
    parser.add_argument("--input", required=True, help="Path to formatted JSONL (output of src.data.format).")
    parser.add_argument("--output", required=True, help="Curriculum-sorted JSONL output path.")
    parser.add_argument("--print-distribution", action="store_true", help="Print difficulty quartile distribution.")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    sorted_examples = sort_by_curriculum(examples)
    scores = [example_difficulty(ex) for ex in sorted_examples]

    with open(out_path, "w") as f:
        for ex in sorted_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Sorted {len(examples)} examples by difficulty -> {out_path}")
    if not scores:
        print("Input was empty - output is an empty JSONL.")
        return
    print(f"Difficulty range: [{scores[0]:.2f}, {scores[-1]:.2f}], median {scores[len(scores)//2]:.2f}")
    if args.print_distribution:
        n = len(scores)
        for q, name in [(0, "min"), (n // 4, "q1"), (n // 2, "median"), (3 * n // 4, "q3"), (n - 1, "max")]:
            print(f"  {name:6s} = {scores[q]:.2f}")


if __name__ == "__main__":
    main()
