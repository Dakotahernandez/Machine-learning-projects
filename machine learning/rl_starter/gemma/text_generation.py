"""
text_generation.py — Text generation with Gemma and various decoding strategies.

Demonstrates greedy, top-k, top-p (nucleus), temperature scaling,
beam search, and repetition penalty. Side-by-side comparison of outputs.

Usage:
    python text_generation.py --prompt "The future of AI is"
    python text_generation.py --prompt "Write a poem about the ocean" --strategy all
    python text_generation.py --model google/gemma-2-9b --prompt "Explain gravity"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("outputs")

STRATEGIES = {
    "greedy": dict(do_sample=False, num_beams=1),
    "top_k": dict(do_sample=True, top_k=50, temperature=0.8),
    "top_p": dict(do_sample=True, top_p=0.92, temperature=0.8),
    "temperature_low": dict(do_sample=True, temperature=0.3, top_p=0.95),
    "temperature_high": dict(do_sample=True, temperature=1.2, top_p=0.95),
    "beam_search": dict(do_sample=False, num_beams=4, early_stopping=True),
}


def load_model(model_name: str, quantize: str | None, device: str):
    print(f"Loading {model_name} ...")
    kwargs: dict = {"device_map": device, "torch_dtype": torch.float16}

    if quantize == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
        kwargs["device_map"] = "auto"
    elif quantize == "int4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)
        kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int, **gen_kwargs) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.15,
            **gen_kwargs,
        )
    elapsed = time.perf_counter() - t0
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma Text Generation")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--strategy", type=str, default="all",
                        choices=list(STRATEGIES.keys()) + ["all"])
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    quant = args.quantize if args.quantize != "none" else None
    device = "auto" if args.device == "auto" else args.device
    model, tokenizer = load_model(args.model, quant, device)

    strategies = STRATEGIES if args.strategy == "all" else {args.strategy: STRATEGIES[args.strategy]}

    print(f"\nPrompt: {args.prompt!r}\n")
    print("=" * 70)

    results = {}
    for name, kwargs in strategies.items():
        text, elapsed = generate(model, tokenizer, args.prompt, args.max_tokens, **kwargs)
        tokens_generated = len(tokenizer.encode(text))
        tps = tokens_generated / elapsed if elapsed > 0 else 0
        results[name] = {"text": text, "time": elapsed, "tokens": tokens_generated, "tps": tps}

        print(f"\n--- {name} ({elapsed:.1f}s, {tps:.0f} tok/s) ---")
        print(text[:500])
        print()

    # save comparison
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_DIR / "generation_comparison.txt", "w", encoding="utf-8") as f:
        f.write(f"Prompt: {args.prompt}\nModel: {args.model}\n\n")
        for name, r in results.items():
            f.write(f"=== {name} ({r['time']:.1f}s, {r['tps']:.0f} tok/s) ===\n")
            f.write(r["text"] + "\n\n")
    print(f"Saved comparison to {OUTPUT_DIR / 'generation_comparison.txt'}")
    print("Done ✓")


if __name__ == "__main__":
    main()
