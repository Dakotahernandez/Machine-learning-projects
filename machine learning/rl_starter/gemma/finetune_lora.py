"""
finetune_lora.py — LoRA fine-tuning Gemma on custom datasets.

Efficiently fine-tunes Gemma with Low-Rank Adaptation,
requiring minimal VRAM (~8 GB for the 2B model).

Features:
  - LoRA adapters on attention layers (r=16, alpha=32)
  - 4-bit quantised base model to save memory
  - Gradient checkpointing for large batches
  - Supports JSONL datasets (instruction/response format)
  - Generates a built-in demo dataset if none supplied
  - Saves merged or adapter-only weights
  - Training loss curve

Usage:
    python finetune_lora.py --epochs 3 --device auto
    python finetune_lora.py --dataset data/my_data.jsonl --epochs 5 --lr 2e-4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

OUTPUT_DIR = Path("outputs")


DEMO_DATA = [
    {"instruction": "What is machine learning?", "response": "Machine learning is a subset of AI where systems learn patterns from data to make predictions without being explicitly programmed."},
    {"instruction": "Explain gradient descent.", "response": "Gradient descent is an optimisation algorithm that iteratively adjusts model parameters by moving in the direction of steepest decrease of the loss function."},
    {"instruction": "What is overfitting?", "response": "Overfitting occurs when a model learns noise in the training data rather than the underlying pattern, leading to poor generalisation on unseen data."},
    {"instruction": "Compare supervised and unsupervised learning.", "response": "Supervised learning uses labelled data to learn input-output mappings. Unsupervised learning finds hidden patterns in unlabelled data, such as clustering or dimensionality reduction."},
    {"instruction": "What is a neural network?", "response": "A neural network is a computational model inspired by biological neurons, consisting of layers of interconnected nodes that transform inputs through weighted connections and activation functions."},
    {"instruction": "Explain regularisation.", "response": "Regularisation adds a penalty term to the loss function to discourage overly complex models. L1 promotes sparsity, L2 penalises large weights, and dropout randomly deactivates neurons during training."},
    {"instruction": "What is transfer learning?", "response": "Transfer learning reuses a model trained on one task as the starting point for a different but related task, drastically reducing the data and compute needed for the new task."},
    {"instruction": "Explain attention mechanism.", "response": "The attention mechanism allows a model to dynamically focus on relevant parts of the input. It computes weighted sums where weights indicate the importance of each input element to the current output."},
    {"instruction": "What is a transformer?", "response": "A transformer is a neural architecture that uses self-attention to process all input positions in parallel, enabling efficient training on long sequences. It forms the backbone of modern LLMs like Gemma."},
    {"instruction": "Explain backpropagation.", "response": "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule layer by layer from output to input, enabling gradient-based optimisation."},
    {"instruction": "What is batch normalisation?", "response": "Batch normalisation normalises layer inputs across the mini-batch to have zero mean and unit variance, stabilising training and allowing higher learning rates."},
    {"instruction": "Explain the bias-variance tradeoff.", "response": "Bias is error from oversimplified models (underfitting). Variance is error from models too sensitive to training data (overfitting). The goal is to balance both for good generalisation."},
]


def load_dataset(path: str | None) -> Dataset:
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    else:
        if path:
            print(f"Dataset not found at {path}, using built-in demo data")
        else:
            print("No dataset specified, using built-in demo data (12 examples)")
        records = DEMO_DATA
    return Dataset.from_list(records)


def format_example(example: dict, tokenizer) -> dict:
    """Format instruction/response pair for causal LM training."""
    text = (
        f"<start_of_turn>user\n{example['instruction']}<end_of_turn>\n"
        f"<start_of_turn>model\n{example['response']}<end_of_turn>"
    )
    tokens = tokenizer(text, truncation=True, max_length=512, padding=False)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma LoRA Fine-tuning")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-merged", action="store_true", help="Save full merged model (large)")
    args = parser.parse_args()

    print(f"Loading {args.model} in 4-bit ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # data
    dataset = load_dataset(args.dataset)
    tokenized = dataset.map(lambda ex: format_example(ex, tokenizer), remove_columns=dataset.column_names)
    print(f"Training examples: {len(tokenized)}")

    # training
    output_path = OUTPUT_DIR / "gemma_lora"
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print(f"\nTraining for {args.epochs} epochs ...\n")
    result = trainer.train()

    # save
    OUTPUT_DIR.mkdir(exist_ok=True)
    adapter_path = OUTPUT_DIR / "gemma_lora_adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved to {adapter_path}")

    if args.save_merged:
        print("Merging adapter into base model ...")
        merged = model.merge_and_unload()
        merged_path = OUTPUT_DIR / "gemma_lora_merged"
        merged.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        print(f"Merged model saved to {merged_path}")

    # loss curve
    if trainer.state.log_history:
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        if losses:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(losses)
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.set_title("LoRA Fine-tuning Loss")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "lora_loss.png", dpi=150)
            plt.close(fig)
            print(f"Saved loss curve to {OUTPUT_DIR / 'lora_loss.png'}")

    # quick test
    print("\n--- Quick test after fine-tuning ---")
    model.eval()
    test_prompt = "<start_of_turn>user\nWhat is a loss function?<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, do_sample=True)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Q: What is a loss function?\nA: {response}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
