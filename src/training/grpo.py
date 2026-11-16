"""GRPO training for function-calling (Variants 4-6).

V4/V5: cold-start from base Qwen2.5-3B-Instruct with QLoRA.
V6: SFT-init from a pre-trained masked adapter via --init-adapter, then
GRPO with curriculum-sorted data.

Uses TRL 0.27.1 GRPOTrainer with custom reward functions.
"""

import argparse
import json
from pathlib import Path

import yaml
from datasets import Dataset

from src.training.rewards import correctness_reward
from src.training.save_utils import assert_save_environment, push_adapter_to_hub


def load_config(config_path: str, overrides: dict | None = None) -> dict:
    """Load YAML config with optional nested overrides."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    return config


def prepare_grpo_dataset(data_path: str | Path, tokenizer) -> Dataset:
    """Load formatted JSONL and create dataset for GRPOTrainer.

    Each example becomes:
    - prompt: pre-formatted string (system + user + tools, via chat template)
    - ground_truth: JSON string of expected tool calls
    """
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))

    prompts = []
    ground_truths = []

    for ex in examples:
        prompt_messages = [m for m in ex["messages"] if m["role"] != "assistant"]
        formatted = tokenizer.apply_chat_template(
            prompt_messages,
            tools=ex.get("tools"),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(formatted)
        ground_truths.append(json.dumps(ex["ground_truth"]))

    return Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config: dict):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_cfg = config["model"]
    quant_cfg = config["quantization"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
    )

    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    attn_implementation = model_cfg.get("attn_implementation", "sdpa")

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    return model


def build_peft_config(config: dict):
    from peft import LoraConfig

    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )


def build_training_args(config: dict, run_name: str, hub_repo_id: str | None = None):
    import os

    from trl import GRPOConfig

    grpo_cfg = config["grpo"]
    train_cfg = config["training"]
    log_cfg = config["logging"]

    hub_kwargs = {}
    if hub_repo_id:
        hub_kwargs = {
            "push_to_hub": True,
            "hub_model_id": hub_repo_id,
            "hub_strategy": "every_save",
            "hub_private_repo": True,
            "hub_token": os.environ["HF_TOKEN"],
        }

    # max_steps overrides num_train_epochs when set (>0). Used by --max-steps
    # CLI flag for smoke tests; -1 keeps epoch-based scheduling.
    max_steps = train_cfg.get("max_steps", -1)

    grpo_kwargs = {
        "num_generations": grpo_cfg["num_generations"],
        "max_completion_length": grpo_cfg["max_completion_length"],
        "beta": grpo_cfg["beta"],
        "loss_type": grpo_cfg["loss_type"],
        "scale_rewards": grpo_cfg["scale_rewards"],
        "temperature": grpo_cfg["temperature"],
    }
    if "mask_truncated_completions" in grpo_cfg:
        grpo_kwargs["mask_truncated_completions"] = grpo_cfg["mask_truncated_completions"]
    if "generation_batch_size" in grpo_cfg:
        grpo_kwargs["generation_batch_size"] = grpo_cfg["generation_batch_size"]

    train_kwargs = {
        "learning_rate": train_cfg["learning_rate"],
        "num_train_epochs": train_cfg["num_train_epochs"],
        "max_steps": max_steps,
        "per_device_train_batch_size": train_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "weight_decay": train_cfg["weight_decay"],
        "max_grad_norm": train_cfg["max_grad_norm"],
        "bf16": train_cfg.get("bf16", False),
        "fp16": train_cfg.get("fp16", False),
        "gradient_checkpointing": train_cfg["gradient_checkpointing"],
        "logging_steps": train_cfg["logging_steps"],
        "save_steps": train_cfg["save_steps"],
    }
    if "lr_scheduler_type" in train_cfg:
        train_kwargs["lr_scheduler_type"] = train_cfg["lr_scheduler_type"]

    return GRPOConfig(
        output_dir=f"{config['output_dir']}/{run_name}",
        run_name=run_name,
        **grpo_kwargs,
        **train_kwargs,
        report_to=log_cfg["report_to"],
        **hub_kwargs,
    )


def train(
    config: dict,
    data_path: str,
    run_name: str,
    hf_repo_id: str | None = None,
    no_push: bool = False,
    init_adapter: str | None = None,
):
    """Run GRPO training.

    init_adapter: optional HF repo id or local path to a PEFT adapter to load
    as starting weights (V6 SFT-init path). When set, GRPOTrainer continues
    training the loaded adapter; build_peft_config is bypassed because the
    adapter's own LoraConfig is the source of truth.
    """
    import wandb
    from trl import GRPOTrainer

    resolved_repo_id = assert_save_environment(hf_repo_id, allow_no_push=no_push)

    wandb.init(project=config["logging"]["wandb_project"], name=run_name)

    tokenizer = load_tokenizer(config["model"]["name"])
    dataset = prepare_grpo_dataset(data_path, tokenizer)
    print(f"Dataset: {len(dataset)} examples")

    model = load_model(config)

    # transformers 5.x dropped `PreTrainedModel.warnings_issued`, but TRL 0.27.1
    # GRPOTrainer.__init__ does `model.warnings_issued["estimate_tokens"] = True`
    # to suppress a FLOPs-estimation warning from base Trainer. Without this
    # attribute the trainer crashes before training starts. Set it on the bare
    # model so PEFT's __getattr__ chain finds it from both wrapping paths
    # (V4/V5: TRL wraps with peft_config; V6: we pre-wrap with PeftModel).
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    if init_adapter is not None:
        from peft import PeftModel
        print(f"Loading SFT-init adapter from {init_adapter}")
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        peft_config = None
    else:
        peft_config = build_peft_config(config)

    training_args = build_training_args(config, run_name, hub_repo_id=resolved_repo_id)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    output_dir = Path(config["output_dir"]) / run_name / "final"
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved to {output_dir}")

    if resolved_repo_id:
        push_adapter_to_hub(output_dir, resolved_repo_id, run_name)

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="GRPO training for function calling")
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--data", required=True, help="Path to formatted JSONL")
    parser.add_argument("--run-name", required=True, help="W&B run name (e.g. grpo-v4)")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap training at N optimizer steps. Use ~5 for end-to-end smoke "
             "tests of rollout, reward, optim, and Hub push before committing "
             "to a multi-hour run. -1 (default) keeps epoch-based scheduling.",
    )
    parser.add_argument("--hf-repo-id", help="HF Hub repo (user/repo). Falls back to HF_REPO_ID env var.")
    parser.add_argument("--no-push", action="store_true", help="Skip HF push. For local smoke tests only.")
    parser.add_argument(
        "--init-adapter",
        help="HF repo id or local path of a PEFT adapter to load as starting weights (V6 SFT-init).",
    )
    args = parser.parse_args()

    overrides = {}
    if args.beta is not None:
        overrides["grpo"] = {"beta": args.beta}
    if args.learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = args.max_steps

    config = load_config(args.config, overrides=overrides if overrides else None)
    train(
        config,
        args.data,
        args.run_name,
        hf_repo_id=args.hf_repo_id,
        no_push=args.no_push,
        init_adapter=args.init_adapter,
    )


if __name__ == "__main__":
    main()
