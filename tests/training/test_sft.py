"""Tests for SFT tokenization and dataset logic."""

import json

import pytest
import torch

from src.training.sft import (
    SFTDataCollator,
    load_and_tokenize_dataset,
    load_config,
    load_tokenizer,
    tokenize_example,
)

SINGLE_CALL_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with access to functions. Use them if required."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "add", "arguments": {"a": 2, "b": 2}}\n</tool_call>'},
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
    "ground_truth": [{"name": "add", "arguments": {"a": 2, "b": 2}}],
}

PARALLEL_CALL_EXAMPLE = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant with access to functions. Use them if required."},
        {"role": "user", "content": "Add 1+2 and multiply 3*4"},
        {
            "role": "assistant",
            "content": (
                '<tool_call>\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n</tool_call>\n'
                '<tool_call>\n{"name": "multiply", "arguments": {"a": 3, "b": 4}}\n</tool_call>'
            ),
        },
    ],
    "tools": [
        {"type": "function", "function": {"name": "add", "description": "Add two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}},
        {"type": "function", "function": {"name": "multiply", "description": "Multiply two numbers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}},
    ],
    "ground_truth": [
        {"name": "add", "arguments": {"a": 1, "b": 2}},
        {"name": "multiply", "arguments": {"a": 3, "b": 4}},
    ],
}


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer("Qwen/Qwen2.5-3B-Instruct")


@pytest.fixture(scope="module")
def single_result(tokenizer):
    return tokenize_example(SINGLE_CALL_EXAMPLE, tokenizer, max_seq_length=2048)


@pytest.fixture(scope="module")
def parallel_result(tokenizer):
    return tokenize_example(PARALLEL_CALL_EXAMPLE, tokenizer, max_seq_length=2048)


def test_returns_required_keys(single_result):
    assert "input_ids" in single_result
    assert "attention_mask" in single_result
    assert "labels" in single_result


def test_labels_mask_non_assistant_tokens(single_result):
    labels = single_result["labels"]
    assert labels[0] == -100
    assert any(l != -100 for l in labels)


def test_assistant_tokens_have_loss(single_result):
    labels = single_result["labels"]
    loss_tokens = [l for l in labels if l != -100]
    assert len(loss_tokens) >= 10


def test_last_real_label_is_im_end(single_result):
    labels = single_result["labels"]
    real_labels = [l for l in labels if l != -100]
    assert real_labels[-1] == 151645


def test_parallel_call_has_more_loss_tokens(single_result, parallel_result):
    single_loss = sum(1 for l in single_result["labels"] if l != -100)
    parallel_loss = sum(1 for l in parallel_result["labels"] if l != -100)
    assert parallel_loss > single_loss


def test_truncation_respects_max_length(tokenizer):
    result = tokenize_example(SINGLE_CALL_EXAMPLE, tokenizer, max_seq_length=128)
    assert len(result["input_ids"]) <= 128


def test_lengths_match(single_result):
    assert len(single_result["input_ids"]) == len(single_result["attention_mask"])
    assert len(single_result["input_ids"]) == len(single_result["labels"])


def test_loads_jsonl_file(tokenizer, tmp_path):
    jsonl_path = tmp_path / "test.jsonl"
    with open(jsonl_path, "w") as f:
        f.write(json.dumps(SINGLE_CALL_EXAMPLE) + "\n")
        f.write(json.dumps(PARALLEL_CALL_EXAMPLE) + "\n")

    dataset = load_and_tokenize_dataset(str(jsonl_path), tokenizer, max_seq_length=2048)
    assert len(dataset) == 2


def test_split_produces_train_and_val(tokenizer, tmp_path):
    jsonl_path = tmp_path / "test_split.jsonl"
    with open(jsonl_path, "w") as f:
        for _ in range(20):
            f.write(json.dumps(SINGLE_CALL_EXAMPLE) + "\n")

    train_ds, val_ds = load_and_tokenize_dataset(
        str(jsonl_path), tokenizer, max_seq_length=2048, val_split=0.1, seed=42
    )
    assert len(train_ds) == 18
    assert len(val_ds) == 2


# --- Config loading tests ---


def test_load_config():
    config = load_config("configs/sft.yaml")
    assert config["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert config["lora_rank"] == 64


def test_load_config_with_overrides():
    config = load_config("configs/sft.yaml", overrides={"num_train_epochs": 1})
    assert config["num_train_epochs"] == 1


# --- Model loading tests (GPU-gated) ---


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_model_has_lora_adapters():
    from src.training.sft import load_model

    config = load_config("configs/sft.yaml")
    model = load_model(config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert 0 < trainable / total < 0.05


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_model_is_quantized():
    from src.training.sft import load_model

    config = load_config("configs/sft.yaml")
    model = load_model(config)
    assert model.config.quantization_config is not None


# --- Data collator tests ---


def _make_example(input_ids, labels):
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        "labels": torch.tensor(labels),
    }


def test_collator_pads_to_longest():
    collator = SFTDataCollator(pad_token_id=151643)
    batch = [
        _make_example([1, 2, 3], [-100, 5, 6]),
        _make_example([1, 2, 3, 4, 5], [-100, -100, 7, 8, 9]),
    ]
    result = collator(batch)
    assert result["input_ids"].shape == (2, 5)


def test_collator_padding_values():
    collator = SFTDataCollator(pad_token_id=151643)
    batch = [
        _make_example([1, 2], [-100, 5]),
        _make_example([1, 2, 3, 4], [-100, -100, 7, 8]),
    ]
    result = collator(batch)
    assert result["input_ids"][0, 2].item() == 151643
    assert result["attention_mask"][0, 2].item() == 0
    assert result["labels"][0, 2].item() == -100
