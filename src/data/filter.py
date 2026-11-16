"""Filter xLAM-60k down to a ~4K high-quality subset for training.

Filtering criteria:
  - Remove examples with malformed JSON or missing fields
  - Remove duplicates (same query text)
  - Balance the distribution across call types (single, parallel, no-call)
  - Prefer examples with more diverse parameter types
  - Cap at target_size (default 4000)

Usage:
    python -m src.data.filter \
        --input data/raw/xlam_60k.json \
        --output data/processed/xlam_4k.json \
        --size 4000
"""

import argparse
import json
import random
from pathlib import Path


def is_valid(example: dict) -> bool:
    """Check that an example has all required fields and is well-formed."""
    if not example.get("query") or not isinstance(example["query"], str):
        return False
    if not isinstance(example.get("tools"), list):
        return False
    if not isinstance(example.get("answers"), list):
        return False

    # Each tool needs name, description, parameters
    for tool in example["tools"]:
        if not all(k in tool for k in ("name", "description", "parameters")):
            return False

    # Each answer needs name and arguments
    for ans in example["answers"]:
        if not all(k in ans for k in ("name", "arguments")):
            return False
        # The called function must exist in the tool list
        tool_names = {t["name"] for t in example["tools"]}
        if ans["name"] not in tool_names:
            return False

    return True


def classify_example(example: dict) -> str:
    """Classify by call type: single, parallel, or no_call."""
    n = len(example["answers"])
    if n == 0:
        return "no_call"
    elif n == 1:
        return "single"
    else:
        return "parallel"


def param_diversity_score(example: dict) -> int:
    """Score based on variety of parameter types used.

    Higher score = more diverse. Prefer examples that exercise
    different parameter types (string, int, float, list, bool).
    """
    types_seen = set()
    for tool in example["tools"]:
        for param_info in tool["parameters"].values():
            if isinstance(param_info, dict):
                types_seen.add(param_info.get("type", "unknown"))
    return len(types_seen)


def filter_dataset(
    examples: list[dict],
    target_size: int = 4000,
    seed: int = 42,
) -> list[dict]:
    """Filter and balance the dataset."""
    random.seed(seed)

    # Step 1: Remove invalid examples
    valid = [ex for ex in examples if is_valid(ex)]
    print(f"Valid examples: {len(valid)} / {len(examples)}")

    # Step 2: Deduplicate by query text
    seen_queries = set()
    deduped = []
    for ex in valid:
        q = ex["query"].strip().lower()
        if q not in seen_queries:
            seen_queries.add(q)
            deduped.append(ex)
    print(f"After dedup: {len(deduped)}")

    # Step 3: Group by call type
    groups: dict[str, list[dict]] = {"single": [], "parallel": [], "no_call": []}
    for ex in deduped:
        groups[classify_example(ex)].append(ex)

    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    # Step 4: Sort each group by param diversity (higher first)
    for k in groups:
        groups[k].sort(key=param_diversity_score, reverse=True)

    # Step 5: Balanced sampling
    # Target distribution: 60% single, 30% parallel, 10% no_call
    # This roughly matches BFCL's emphasis on multi-step tasks
    # while keeping enough simple examples for curriculum training.
    allocations = {
        "single": int(target_size * 0.60),
        "parallel": int(target_size * 0.30),
        "no_call": int(target_size * 0.10),
    }

    selected = []
    for k, alloc in allocations.items():
        pool = groups[k]
        n = min(alloc, len(pool))
        selected.extend(pool[:n])

    # If we're short of target_size, fill from the largest pool
    remaining = target_size - len(selected)
    if remaining > 0:
        all_unused = []
        selected_set = {id(ex) for ex in selected}
        for ex in deduped:
            if id(ex) not in selected_set:
                all_unused.append(ex)
        random.shuffle(all_unused)
        selected.extend(all_unused[:remaining])

    random.shuffle(selected)
    print(f"\nFinal dataset: {len(selected)} examples")

    # Print final distribution
    final_dist = {"single": 0, "parallel": 0, "no_call": 0}
    for ex in selected:
        final_dist[classify_example(ex)] += 1
    for k, v in final_dist.items():
        pct = 100 * v / len(selected)
        print(f"  {k}: {v} ({pct:.1f}%)")

    return selected


def main():
    parser = argparse.ArgumentParser(description="Filter xLAM-60k to training subset")
    parser.add_argument("--input", default="data/raw/xlam_60k.json")
    parser.add_argument("--output", default="data/processed/xlam_4k.json")
    parser.add_argument("--size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        examples = json.load(f)

    filtered = filter_dataset(examples, target_size=args.size, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(filtered, f, indent=2)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
