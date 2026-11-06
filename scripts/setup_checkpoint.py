"""
Automated Checkpoint Setup & Verification Script
Extracts fine-tuned Mistral-7B LoRA adapter weights (from Colab zip or local directory),
validates adapter configuration, and configures .env for local RAG inference.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path


def setup_lora_checkpoint(zip_path: str = None, target_dir: str = "checkpoints/mistral_7b_financial_lora"):
    target_path = Path(target_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Look for zip file in common locations if not explicitly provided
    if not zip_path:
        search_paths = [
            Path("mistral_7b_financial_lora.zip"),
            Path("checkpoints/mistral_7b_financial_lora.zip"),
            Path.home() / "Downloads" / "mistral_7b_financial_lora.zip",
        ]
        for p in search_paths:
            if p.exists():
                zip_path = str(p)
                break

    if zip_path and os.path.exists(zip_path):
        print(f"Found checkpoint archive at: {zip_path}")
        print(f"Extracting to {target_path}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("checkpoints")
        print("Extraction complete.")
    elif target_path.exists() and any(target_path.iterdir()):
        print(f"Found existing extracted checkpoint directory at {target_path}")
    else:
        print("\n" + "=" * 60)
        print("Waiting for Colab training to finish.")
        print("Once Cell 8 downloads 'mistral_7b_financial_lora.zip', run:")
        print("  python scripts/setup_checkpoint.py --zip /path/to/mistral_7b_financial_lora.zip")
        print("=" * 60 + "\n")
        return False

    # 2. Validate adapter files
    adapter_config = target_path / "adapter_config.json"
    adapter_weights = target_path / "adapter_model.safetensors"
    adapter_bin = target_path / "adapter_model.bin"

    if not adapter_config.exists():
        # Check if extracted inside a subfolder
        nested = list(target_path.glob("**/adapter_config.json"))
        if nested:
            actual_folder = nested[0].parent
            for item in actual_folder.iterdir():
                shutil.move(str(item), str(target_path))
            shutil.rmtree(actual_folder, ignore_errors=True)

    if (target_path / "adapter_config.json").exists() and ((target_path / "adapter_model.safetensors").exists() or (target_path / "adapter_model.bin").exists()):
        print(f"SUCCESS: Validated fine-tuned adapter weights in {target_path}!")

        # 3. Update .env to activate fine-tuned model
        env_file = Path(".env")
        lines = []
        if env_file.exists():
            with open(env_file, "r") as f:
                lines = f.readlines()

        new_lines = []
        has_provider = False
        has_path = False
        for line in lines:
            if line.startswith("LLM_PROVIDER="):
                new_lines.append("LLM_PROVIDER=local-lora\n")
                has_provider = True
            elif line.startswith("LORA_ADAPTER_PATH="):
                new_lines.append(f"LORA_ADAPTER_PATH={target_path}\n")
                has_path = True
            else:
                new_lines.append(line)

        if not has_provider:
            new_lines.append("LLM_PROVIDER=local-lora\n")
        if not has_path:
            new_lines.append(f"LORA_ADAPTER_PATH={target_path}\n")

        with open(env_file, "w") as f:
            f.writelines(new_lines)

        print(f"Updated .env: LLM_PROVIDER=local-lora, LORA_ADAPTER_PATH={target_path}")
        print("Your RAG pipeline and Streamlit dashboard are now powered by your fine-tuned Mistral-7B model!")
        return True
    else:
        print(f"Warning: Expected adapter files not found in {target_path}.")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup and verify fine-tuned LoRA checkpoint")
    parser.add_argument("--zip", default=None, help="Path to mistral_7b_financial_lora.zip")
    parser.add_argument("--dir", default="checkpoints/mistral_7b_financial_lora", help="Target checkpoint dir")
    args = parser.parse_args()

    setup_lora_checkpoint(zip_path=args.zip, target_dir=args.dir)
