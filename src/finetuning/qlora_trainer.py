"""
QLoRA Fine-Tuning Pipeline for Mistral-7B-Instruct-v0.3
Designed for training on NVIDIA GPUs (e.g., T4/A10G/A100 via Colab or RunPod).
Features:
- 4-bit NormalFloat (NF4) double quantization
- LoRA targeting attention projections (q, k, v, o)
- Loss masking to backpropagate exclusively over assistant tokens
"""

import argparse
import os
import sys
from typing import Optional


def check_gpu_environment() -> bool:
    """Validate CUDA availability and GPU memory requirements."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("\n" + "=" * 70)
            print("WARNING: CUDA is not available on this machine (Host OS: macOS/CPU).")
            print("QLoRA 4-bit training with bitsandbytes requires an NVIDIA GPU (T4, A10G, A100).")
            print("To train Mistral-7B on Google Colab or a cloud instance, run:")
            print("  pip install -r requirements-train.txt")
            print("  python src/finetuning/qlora_trainer.py --train_file data/finetune_train.jsonl")
            print("=" * 70 + "\n")
            return False
        return True
    except ImportError:
        print("PyTorch is not installed in the current environment.")
        return False


def run_qlora_training(
    train_file: str = "data/finetune_train.jsonl",
    val_file: Optional[str] = "data/finetune_val.jsonl",
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_dir: str = "checkpoints/mistral_7b_financial_lora",
    num_epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Execute complete QLoRA training workflow with 4-bit NF4 quantization."""
    if not check_gpu_environment():
        return

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    print(f"Loading base model '{base_model_name}' in 4-bit NF4...")

    # Configure 4-bit NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # Configure LoRA adapters
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load dataset
    data_files = {"train": train_file}
    if val_file and os.path.exists(val_file):
        data_files["validation"] = val_file
    dataset = load_dataset("json", data_files=data_files)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if "validation" in dataset else "no",
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=2048,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Beginning QLoRA training run...")
    trainer.train()

    print(f"Saving fine-tuned adapter weights to '{output_dir}'...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train_file", default="data/finetune_train.jsonl")
    parser.add_argument("--val_file", default="data/finetune_val.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/mistral_7b_financial_lora")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run_qlora_training(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
    )

