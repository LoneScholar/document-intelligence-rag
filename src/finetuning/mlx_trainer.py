"""
Apple Silicon (MLX) Local LoRA Fine-Tuning for Mistral-7B
Runs natively on macOS with M1/M2/M3/M4 unified memory without NVIDIA CUDA.
Uses mlx-lm to fine-tune Mistral-7B directly on Apple Metal GPU.
"""

import argparse
import json
import os
from pathlib import Path


def format_dataset_for_mlx(train_file: str, output_dir: str = "data/mlx_data"):
    """
    MLX-LM expects train.jsonl and valid.jsonl with {"text": "..."} format
    or standard ChatML messages.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    formatted_samples = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            # Reconstruct ChatML text prompt
            messages = item.get("messages", [])
            text_blocks = []
            for m in messages:
                role = m["role"]
                content = m["content"]
                text_blocks.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            formatted_samples.append({"text": "\n".join(text_blocks)})

    # Split 85% train, 15% validation
    split_idx = int(len(formatted_samples) * 0.85)
    train_samples = formatted_samples[:split_idx]
    val_samples = formatted_samples[split_idx:] if split_idx < len(formatted_samples) else formatted_samples

    with open(out_path / "train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(out_path / "valid.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s) + "\n")

    print(f"Formatted dataset for MLX in {output_dir}: {len(train_samples)} train, {len(val_samples)} val.")
    return str(out_path)


def run_mlx_lora(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    data_dir: str = "data/mlx_data",
    adapter_path: str = "checkpoints/mistral_7b_mlx_lora",
    iters: int = 200,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
):
    """
    Launch native Apple MLX LoRA fine-tuning on macOS.
    Requires: pip install mlx-lm
    """
    try:
        import mlx.core as mx
        print(f"Apple MLX detected! Device: Metal GPU (Unified Memory).")
    except ImportError:
        print("MLX is not installed. To install Apple MLX for native macOS training, run:")
        print("  pip install mlx-lm")
        return

    cmd = (
        f"python -m mlx_lm.lora "
        f"--model {model_name} "
        f"--data {data_dir} "
        f"--train "
        f"--batch-size {batch_size} "
        f"--lora-layers 16 "
        f"--iters {iters} "
        f"--learning-rate {learning_rate} "
        f"--adapter-path {adapter_path}"
    )
    print(f"Executing native Apple Metal LoRA command:\n  {cmd}")
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apple Silicon MLX Fine-Tuning for Mistral-7B")
    parser.add_argument("--train_file", default="data/finetune_train.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/mistral_7b_mlx_lora")
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    data_dir = format_dataset_for_mlx(args.train_file)
    run_mlx_lora(data_dir=data_dir, adapter_path=args.output_dir, iters=args.iters)
