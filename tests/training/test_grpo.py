# tests/training/test_grpo.py
import json

import pytest

from src.training.grpo import prepare_grpo_dataset, load_config


EXAMPLE_FORMATTED = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user", "content": "What is 2 + 3?"},
        {
            "role": "assistant",
            "content": '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 3}}\n</tool_call>',
        },
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer", "description": "First number"},
                        "b": {"type": "integer", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ],
    "ground_truth": [{"name": "add", "arguments": {"a": 2, "b": 3}}],
}


@pytest.fixture
def sample_jsonl(tmp_path):
    path = tmp_path / "data.jsonl"
    with open(path, "w") as f:
        for _ in range(3):
            f.write(json.dumps(EXAMPLE_FORMATTED) + "\n")
    return path


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


class TestPrepareDataset:
    def test_returns_dataset_with_required_columns(self, sample_jsonl, tokenizer):
        ds = prepare_grpo_dataset(sample_jsonl, tokenizer)
        assert "prompt" in ds.column_names
        assert "ground_truth" in ds.column_names

    def test_prompt_is_string(self, sample_jsonl, tokenizer):
        ds = prepare_grpo_dataset(sample_jsonl, tokenizer)
        assert isinstance(ds[0]["prompt"], str)

    def test_prompt_excludes_assistant_turn(self, sample_jsonl, tokenizer):
        # The assistant's actual response content should not appear in the prompt.
        # Note: Qwen's chat template includes <tool_call> in system instructions,
        # so we check for the specific assistant payload rather than the tag itself.
        ds = prepare_grpo_dataset(sample_jsonl, tokenizer)
        assert '"a": 2, "b": 3' not in ds[0]["prompt"]

    def test_prompt_includes_tool_definitions(self, sample_jsonl, tokenizer):
        ds = prepare_grpo_dataset(sample_jsonl, tokenizer)
        assert "add" in ds[0]["prompt"]
        assert "Add two numbers" in ds[0]["prompt"]

    def test_ground_truth_is_json_string(self, sample_jsonl, tokenizer):
        ds = prepare_grpo_dataset(sample_jsonl, tokenizer)
        parsed = json.loads(ds[0]["ground_truth"])
        assert isinstance(parsed, list)
        assert parsed[0]["name"] == "add"

    def test_dataset_length(self, sample_jsonl, tokenizer):
        ds = prepare_grpo_dataset(sample_jsonl, tokenizer)
        assert len(ds) == 3


class TestLoadConfig:
    def test_loads_grpo_config(self):
        config = load_config("configs/grpo.yaml")
        assert config["model"]["name"] == "Qwen/Qwen2.5-3B-Instruct"
        assert config["grpo"]["num_generations"] == 4
        assert config["grpo"]["beta"] == 0.04

    def test_override(self):
        config = load_config("configs/grpo.yaml", overrides={"grpo": {"beta": 0.1}})
        assert config["grpo"]["beta"] == 0.1
