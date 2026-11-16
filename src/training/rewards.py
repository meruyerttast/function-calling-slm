"""Reward functions for GRPO function-calling training.

All functions are pure (no model dependency) and return list[float].
Signature follows TRL 0.27.1 GRPOTrainer convention:
reward_func(completions=..., **dataset_columns) -> list[float]
"""

import json
import re


def parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from <tool_call>...</tool_call> blocks."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    for match in matches:
        try:
            calls.append(json.loads(match))
        except json.JSONDecodeError:
            continue
    return calls


def correctness_reward(completions: list[str], ground_truth: list[str], **kwargs) -> list[float]:
    """1.0 if function name AND all param key-value pairs match, else 0.0.

    Comparison is order-invariant for parallel calls (sorted by function name).
    ground_truth entries are JSON strings of list[dict] with "name" and "arguments".
    """
    rewards = []
    for completion, truth_str in zip(completions, ground_truth):
        pred_calls = parse_tool_calls(completion)
        true_calls = json.loads(truth_str) if isinstance(truth_str, str) else truth_str

        if len(pred_calls) != len(true_calls):
            rewards.append(0.0)
            continue

        def _sort_key(call):
            return (call.get("name", ""), json.dumps(call.get("arguments", {}), sort_keys=True))

        pred_sorted = sorted(pred_calls, key=_sort_key)
        true_sorted = sorted(true_calls, key=_sort_key)

        match = all(
            p.get("name") == t.get("name") and p.get("arguments") == t.get("arguments")
            for p, t in zip(pred_sorted, true_sorted)
        )
        rewards.append(1.0 if match else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    """0.5 if output contains valid <tool_call>...</tool_call> tags, else 0.0."""
    rewards = []
    for completion in completions:
        has_tags = bool(re.search(r"<tool_call>.*?</tool_call>", completion, re.DOTALL))
        rewards.append(0.5 if has_tags else 0.0)
    return rewards
