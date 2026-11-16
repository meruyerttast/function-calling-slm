"""Apply Hammer-style function masking to training data.

Replaces function names, parameter names, and default values with
random strings. Forces the model to reason from descriptions instead
of memorizing naming conventions.

Based on: Hammer 2.1 (ICLR 2025), arXiv:2410.04587
Reference impl: github.com/MadeAgents/Hammer/blob/main/train/data_processing.py

Masking is applied to ~67% of examples. The other ~33% keep original names.

Usage:
    python -m src.data.mask \
        --input data/processed/xlam_4k.json \
        --output data/processed/xlam_4k_masked.json
"""

import argparse
import copy
import json
import random
import string
from pathlib import Path

# Character set for generating random names.
# Matches Hammer: uppercase + lowercase + underscore + period + digits
CHARSET = (
    list(string.ascii_uppercase)
    + list(string.ascii_lowercase)
    + ["_", "."]
    + list(string.digits)
)


def random_name(min_len: int = 5, max_len: int = 15) -> str:
    """Generate a random identifier string."""
    length = random.randint(min_len, max_len)
    # First character should be a letter (valid identifier start)
    first = random.choice(string.ascii_letters)
    rest = "".join(random.choice(CHARSET) for _ in range(length - 1))
    return first + rest


def mask_function_names(example: dict) -> dict:
    """Replace all function names with random strings.

    Updates both the tool definitions and the answer keys
    so they stay consistent.
    """
    name_map = {}
    for tool in example["tools"]:
        original = tool["name"]
        if original not in name_map:
            name_map[original] = random_name(5, 15)
        tool["name"] = name_map[original]

    for ans in example["answers"]:
        original = ans["name"]
        if original in name_map:
            ans["name"] = name_map[original]

    return example


def mask_param_names(example: dict) -> dict:
    """Replace all parameter names with random strings.

    Each parameter gets a unique masked name within its function.
    The same mapping is applied to the answer arguments.
    """
    # Build per-function param mappings
    func_param_maps: dict[str, dict[str, str]] = {}

    for tool in example["tools"]:
        fname = tool["name"]
        param_map = {}
        new_params = {}

        for param_name, param_info in tool["parameters"].items():
            masked = random_name(4, 10)
            param_map[param_name] = masked
            new_params[masked] = param_info

        tool["parameters"] = new_params
        func_param_maps[fname] = param_map

    # Update answer arguments to use masked param names
    for ans in example["answers"]:
        fname = ans["name"]
        if fname in func_param_maps:
            param_map = func_param_maps[fname]
            new_args = {}
            for arg_name, arg_val in ans["arguments"].items():
                masked_name = param_map.get(arg_name, arg_name)
                new_args[masked_name] = arg_val
            ans["arguments"] = new_args

    return example


def mask_default_values(example: dict) -> dict:
    """Replace default parameter values with random alternatives.

    - Numeric defaults get shifted by a small random offset
    - String defaults get replaced with random strings
    - Booleans get flipped
    """
    for tool in example["tools"]:
        for param_info in tool["parameters"].values():
            if not isinstance(param_info, dict):
                continue
            if "default" not in param_info:
                continue

            default = param_info["default"]
            if isinstance(default, bool):
                param_info["default"] = not default
            elif isinstance(default, int):
                param_info["default"] = default + random.randint(1, 4)
            elif isinstance(default, float):
                param_info["default"] = round(default + random.uniform(0.5, 3.0), 2)
            elif isinstance(default, str) and default:
                param_info["default"] = random_name(5, 5)

    return example


def mask_example(example: dict) -> dict:
    """Apply all three masking stages to a single example."""
    ex = copy.deepcopy(example)
    ex = mask_function_names(ex)
    ex = mask_param_names(ex)
    ex = mask_default_values(ex)
    return ex


def mask_dataset(
    examples: list[dict],
    mask_ratio: float = 0.67,
    seed: int = 42,
) -> list[dict]:
    """Apply function masking to a fraction of the dataset.

    Args:
        examples: Raw training examples
        mask_ratio: Fraction to mask (default 0.67, matching Hammer)
        seed: Random seed for reproducibility

    Returns:
        Combined list of masked + unmasked examples, shuffled
    """
    random.seed(seed)

    masked_count = 0
    result = []

    for ex in examples:
        if random.random() < mask_ratio:
            result.append(mask_example(ex))
            masked_count += 1
        else:
            result.append(copy.deepcopy(ex))

    random.shuffle(result)

    print(f"Masked {masked_count} / {len(examples)} examples ({100 * masked_count / len(examples):.0f}%)")
    print(f"Unmasked: {len(examples) - masked_count}")
    print(f"Total output: {len(result)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Apply function masking")
    parser.add_argument("--input", default="data/processed/xlam_4k.json")
    parser.add_argument("--output", default="data/processed/xlam_4k_masked.json")
    parser.add_argument("--mask-ratio", type=float, default=0.67)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        examples = json.load(f)

    masked = mask_dataset(examples, mask_ratio=args.mask_ratio, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(masked, f, indent=2)

    print(f"Saved to {args.output}")

    # Print a sample to show masking worked
    original_names = set()
    with open(args.input) as f:
        for ex in json.load(f):
            for t in ex["tools"]:
                original_names.add(t["name"])

    for ex in masked:
        if ex["answers"] and ex["answers"][0]["name"] not in original_names:
            print(f"\nSample masked example:")
            print(f"  Query: {ex['query'][:80]}...")
            print(f"  Tools: {[t['name'] for t in ex['tools']]}")
            print(f"  Answer: {ex['answers'][0]['name']}({list(ex['answers'][0]['arguments'].keys())})")
            break


if __name__ == "__main__":
    main()
