"""SFT dataset and tokenization for function-calling fine-tuning.

Converts JSONL examples (from the data pipeline) into pre-tokenized tensors
with loss masking: only assistant-turn tokens receive gradient signal.

Usage:
    from src.training.sft import load_tokenizer, load_and_tokenize_dataset

    tokenizer = load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
    train_ds, val_ds = load_and_tokenize_dataset(
        "data/processed/sft_train.jsonl", tokenizer, max_seq_length=2048, val_split=0.1
    )
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.training.save_utils import assert_save_environment, push_adapter_to_hub

IGNORE_INDEX = -100


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    return tokenizer


def find_subsequence(seq: list[int], subseq: list[int]) -> int:
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i : i + m] == subseq:
            return i
    return -1


def tokenize_example(example: dict, tokenizer: AutoTokenizer, max_seq_length: int) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"], tools=example["tools"], tokenize=False
    )
    encoded = tokenizer(text, truncation=True, max_length=max_seq_length)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    assistant_marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    marker_pos = find_subsequence(input_ids, assistant_marker)

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    labels = [IGNORE_INDEX] * len(input_ids)
    if marker_pos != -1:
        assistant_start = marker_pos + len(assistant_marker)
        for i in range(assistant_start, len(input_ids)):
            labels[i] = input_ids[i]
            if input_ids[i] == im_end_id:
                break

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class SFTDataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(ex["labels"], dtype=torch.long),
        }


def load_and_tokenize_dataset(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    val_split: float = 0.0,
    seed: int = 42,
) -> SFTDataset | tuple[SFTDataset, SFTDataset]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the data pipeline first (see README)."
        )

    with open(path) as f:
        raw_examples = [json.loads(line) for line in f if line.strip()]

    tokenized = []
    truncated = 0
    for ex in raw_examples:
        result = tokenize_example(ex, tokenizer, max_seq_length)
        if len(result["input_ids"]) == max_seq_length:
            truncated += 1
        tokenized.append(result)

    print(f"Tokenized {len(tokenized)} examples")
    if truncated:
        print(f"  {truncated} examples truncated to {max_seq_length} tokens")

    if val_split <= 0.0:
        return SFTDataset(tokenized)

    rng = random.Random(seed)
    indices = list(range(len(tokenized)))
    rng.shuffle(indices)

    val_size = int(len(tokenized) * val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_examples = [tokenized[i] for i in train_indices]
    val_examples = [tokenized[i] for i in val_indices]

    print(f"  Train: {len(train_examples)}, Val: {len(val_examples)}")
    return SFTDataset(train_examples), SFTDataset(val_examples)


def load_config(config_path: str, overrides: dict | None = None) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        config.update(overrides)
    return config


def load_model(config: dict):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    compute_dtype = torch.bfloat16 if config.get("bf16", False) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["lora_target_modules"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


@dataclass
class SFTDataCollator:
    pad_token_id: int

    def _pad(self, tensor: torch.Tensor, length: int, value: int) -> torch.Tensor:
        pad = torch.full((length,), value, dtype=tensor.dtype)
        return torch.cat([tensor, pad])

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)

        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(self._pad(ex["input_ids"], pad_len, self.pad_token_id))
            attention_mask.append(self._pad(ex["attention_mask"], pad_len, 0))
            labels.append(self._pad(ex["labels"], pad_len, IGNORE_INDEX))

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def train(config: dict, data_path: str, run_name: str, hf_repo_id: str | None = None, no_push: bool = False):
    resolved_repo_id = assert_save_environment(hf_repo_id, allow_no_push=no_push)

    if config.get("report_to") == "wandb":
        try:
            import wandb
            os.environ.setdefault("WANDB_PROJECT", "info290-function-calling-slm")
            wandb.init(name=run_name, config=config)
        except ImportError:
            print("Warning: wandb not installed, logging disabled.")
            config["report_to"] = "none"

    tokenizer = load_tokenizer(config["model_name"])
    model = load_model(config)

    val_split = config.get("val_split", 0.1)
    seed = config.get("seed", 42)

    result = load_and_tokenize_dataset(data_path, tokenizer, config["max_seq_length"], val_split, seed)
    if isinstance(result, tuple):
        train_ds, val_ds = result
    else:
        train_ds, val_ds = result, None

    output_dir = os.path.join(config["output_dir"], run_name)

    hub_kwargs = {}
    if resolved_repo_id:
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": resolved_repo_id,
            "hub_strategy": "every_save",
            "hub_private_repo": True,
            "hub_token": os.environ["HF_TOKEN"],
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        optim=config["optim"],
        bf16=config.get("bf16", False),
        fp16=config.get("fp16", False),
        logging_steps=config["logging_steps"],
        eval_strategy=config.get("eval_strategy", "epoch") if val_ds else "no",
        save_strategy=config["save_strategy"],
        seed=seed,
        report_to=config.get("report_to", "none"),
        run_name=run_name,
        remove_unused_columns=False,
        **hub_kwargs,
    )

    collator = SFTDataCollator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved adapter to {final_dir}")

    if resolved_repo_id:
        push_adapter_to_hub(final_dir, resolved_repo_id, run_name)


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT for Qwen2.5-3B function calling")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-train-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hf-repo-id", help="HF Hub repo (user/repo). Falls back to HF_REPO_ID env var.")
    parser.add_argument("--no-push", action="store_true", help="Skip HF push. For local smoke tests only.")
    args = parser.parse_args()

    overrides = {}
    if args.num_train_epochs is not None:
        overrides["num_train_epochs"] = args.num_train_epochs
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    config = load_config(args.config, overrides)
    train(config, args.data, args.run_name, hf_repo_id=args.hf_repo_id, no_push=args.no_push)


if __name__ == "__main__":
    main()
