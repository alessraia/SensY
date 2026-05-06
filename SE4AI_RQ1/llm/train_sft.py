import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

from llm.config import (
    MODEL_CONFIGS,
    TEMP_DIR,
    RANDOM_SEED,
    MAX_SEQ_LENGTH,
    NUM_TRAIN_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    PER_DEVICE_TRAIN_BATCH_SIZE,
    PER_DEVICE_EVAL_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LOGGING_STEPS,
    SAVE_TOTAL_LIMIT,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_bf16_flag() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def load_json_dataset(path: Path, max_samples: int | None = None):
    ds = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.select(range(max_samples))
    return ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gemma", "llama"], required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-dev-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_TRAIN_EPOCHS)
    args = parser.parse_args()

    seed_everything(RANDOM_SEED)

    model_cfg = MODEL_CONFIGS[args.model]
    model_id = model_cfg["model_id"]
    output_dir = model_cfg["output_dir"]
    logs_dir = model_cfg["logs_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    train_path = TEMP_DIR / "sensy_train_sft.jsonl"
    dev_path = TEMP_DIR / "sensy_dev_sft.jsonl"

    print(f"Loading train dataset from: {train_path}")
    print(f"Loading dev dataset from:   {dev_path}")

    train_dataset = load_json_dataset(train_path, args.max_train_samples)
    dev_dataset = load_json_dataset(dev_path, args.max_dev_samples)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples:   {len(dev_dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if get_bf16_flag() else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if get_bf16_flag() else torch.float16,
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    bf16 = get_bf16_flag()

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logs_dir),
        num_train_epochs=args.epochs,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=not bf16,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,
        report_to="tensorboard",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=RANDOM_SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print(f"Starting training for model: {args.model} ({model_id})")
    trainer.train()

    print("Saving final adapter and tokenizer...")
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))

    print("Done.")
    print(f"Final adapter saved in: {output_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()