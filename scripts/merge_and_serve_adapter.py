"""
Merge LoRA Adapters & Export Utility
Merges trained LoRA adapter weights back into Mistral-7B-Instruct-v0.3
and generates an Ollama Modelfile for local serving.
"""

import argparse
import os
import sys
from pathlib import Path


def generate_ollama_modelfile(output_dir: str, model_tag: str = "mistral-financial"):
    """Create Ollama Modelfile for one-command local serving."""
    modelfile_content = f"""FROM mistral:7b-instruct

PARAMETER temperature 0.1
PARAMETER top_p 0.95
PARAMETER stop "<|im_end|>"

SYSTEM \"\"\"You are an elite, SEC-compliant Financial Document Intelligence AI Assistant.
Rely strictly on the provided SEC 10-K excerpts. Every factual assertion must be followed by its bracketed citation anchor [Ticker, Year, Section]. If disclosures are insufficient, refuse.\"\"\"
"""
    modelfile_path = Path(output_dir) / "Modelfile"
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"\nCreated Ollama Modelfile at {modelfile_path}")
    print(f"To serve with Ollama, run:")
    print(f"  ollama create {model_tag} -f {modelfile_path}")
    print(f"  ollama run {model_tag}\n")


def merge_lora_weights(
    base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    adapter_path: str = "checkpoints/mistral_7b_financial_lora",
    output_dir: str = "checkpoints/mistral_7b_financial_merged",
):
    """Merge LoRA adapter into base model weights."""
    if not os.path.exists(adapter_path):
        print(f"Adapter path '{adapter_path}' does not exist.")
        print("Once Colab training completes, place the checkpoint folder in ./checkpoints/")
        generate_ollama_modelfile(output_dir="checkpoints")
        return

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading base model '{base_model_name}'...")
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        print(f"Loading LoRA adapter from '{adapter_path}'...")
        model = PeftModel.from_pretrained(base_model, adapter_path)

        print("Merging adapter weights into base model (merge_and_unload)...")
        merged_model = model.merge_and_unload()

        print(f"Saving merged standalone model to '{output_dir}'...")
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        generate_ollama_modelfile(output_dir)
        print("Merge complete! Standalone fine-tuned model ready for inference.")

    except Exception as e:
        print(f"Could not merge locally ({e}). You can merge directly on Colab or use Ollama.")
        generate_ollama_modelfile(output_dir="checkpoints")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA Adapters & Generate Ollama Modelfile")
    parser.add_argument("--adapter_path", default="checkpoints/mistral_7b_financial_lora")
    parser.add_argument("--output_dir", default="checkpoints/mistral_7b_financial_merged")
    args = parser.parse_args()

    merge_lora_weights(adapter_path=args.adapter_path, output_dir=args.output_dir)
