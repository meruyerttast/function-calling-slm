# tests/training/test_rewards.py
import pytest
from src.training.rewards import parse_tool_calls, correctness_reward, format_reward


class TestParseToolCalls:
    def test_single_tool_call(self):
        text = '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>'
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "add"
        assert result[0]["arguments"] == {"a": 2, "b": 3}

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>\n'
            '<tool_call>\n{"name": "multiply", "arguments": {"x": 3, "y": 4}}\n</tool_call>'
        )
        result = parse_tool_calls(text)
        assert len(result) == 2
        assert result[0]["name"] == "add"
        assert result[1]["name"] == "multiply"

    def test_no_tool_calls(self):
        text = "I cannot help with that request."
        result = parse_tool_calls(text)
        assert result == []

    def test_malformed_json_skipped(self):
        text = '<tool_call>\n{not valid json}\n</tool_call>'
        result = parse_tool_calls(text)
        assert result == []

    def test_mixed_valid_and_invalid(self):
        text = (
            '<tool_call>\n{"name": "add", "arguments": {"a": 1}}\n</tool_call>\n'
            '<tool_call>\n{broken}\n</tool_call>'
        )
        result = parse_tool_calls(text)
        assert len(result) == 1
        assert result[0]["name"] == "add"


class TestCorrectnessReward:
    def test_exact_match(self):
        completions = ['<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>']
        ground_truth = ['[{"name": "add", "arguments": {"a": 2, "b": 3}}]']
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [1.0]

    def test_wrong_function_name(self):
        completions = ['<tool_call>\n{"name": "subtract", "arguments": {"a": 2, "b": 3}}\n</tool_call>']
        ground_truth = ['[{"name": "add", "arguments": {"a": 2, "b": 3}}]']
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [0.0]

    def test_wrong_arguments(self):
        completions = ['<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 99}}\n</tool_call>']
        ground_truth = ['[{"name": "add", "arguments": {"a": 2, "b": 3}}]']
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [0.0]

    def test_missing_call(self):
        completions = ["I don't know how to do that."]
        ground_truth = ['[{"name": "add", "arguments": {"a": 2, "b": 3}}]']
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [0.0]

    def test_parallel_calls_order_invariant(self):
        completions = [
            '<tool_call>\n{"name": "multiply", "arguments": {"x": 3, "y": 4}}\n</tool_call>\n'
            '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>'
        ]
        ground_truth = [
            '[{"name": "add", "arguments": {"a": 1, "b": 2}}, '
            '{"name": "multiply", "arguments": {"x": 3, "y": 4}}]'
        ]
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [1.0]

    def test_wrong_call_count(self):
        completions = ['<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>']
        ground_truth = [
            '[{"name": "add", "arguments": {"a": 1, "b": 2}}, '
            '{"name": "multiply", "arguments": {"x": 3, "y": 4}}]'
        ]
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [0.0]

    def test_duplicate_function_names_different_args(self):
        completions = [
            '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>\n'
            '<tool_call>\n{"name": "add", "arguments": {"a": 3, "b": 4}}\n</tool_call>'
        ]
        ground_truth = [
            '[{"name": "add", "arguments": {"a": 3, "b": 4}}, '
            '{"name": "add", "arguments": {"a": 1, "b": 2}}]'
        ]
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [1.0]

    def test_batch_of_multiple(self):
        completions = [
            '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>',
            "No tool call here.",
        ]
        ground_truth = [
            '[{"name": "add", "arguments": {"a": 1, "b": 2}}]',
            '[{"name": "sub", "arguments": {"a": 5, "b": 3}}]',
        ]
        assert correctness_reward(completions=completions, ground_truth=ground_truth) == [1.0, 0.0]


class TestFormatReward:
    def test_valid_format(self):
        completions = ['<tool_call>\n{"name": "add", "arguments": {"a": 1}}\n</tool_call>']
        assert format_reward(completions=completions) == [0.5]

    def test_no_tags(self):
        completions = ["Here is a plain text response."]
        assert format_reward(completions=completions) == [0.0]

    def test_partial_tags(self):
        completions = ['<tool_call>{"name": "add"}']
        assert format_reward(completions=completions) == [0.0]

    def test_batch(self):
        completions = [
            '<tool_call>\n{"name": "add", "arguments": {}}\n</tool_call>',
            "No tools.",
            '<tool_call>\n{"name": "sub", "arguments": {}}\n</tool_call>',
        ]
        assert format_reward(completions=completions) == [0.5, 0.0, 0.5]
