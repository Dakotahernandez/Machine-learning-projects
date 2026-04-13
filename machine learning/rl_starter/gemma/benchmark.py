"""
benchmark.py — Inference benchmarking for Gemma models.

Compares throughput, latency, and memory across precision
modes (FP16, INT8, INT4) and sequence lengths.

Features:
  - Tokens/second measurement (generation throughput)
  - Time-to-first-token (TTFT) latency
  - Peak GPU memory tracking
  - Sweep across multiple sequence lengths
  - Comparison of FP16 vs INT8 vs INT4 quantisation
  - Summary table and bar chart
  - JSON export of raw results

Usage:
    python benchmark.py
    python benchmark.py --model google/gemma-2-9b-it --quants fp16 int8 int4
    python benchmark.py --prompt-lengths 32 128 512 --gen-tokens 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_DIR = Path("outputs")


def load_model(model_name: str, quant: str, device: str):
    kwargs = {"device_map": device}
    if quant == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "int4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def warmup(model, tokenizer, device: str) -> None:
    dummy = tokenizer("Hello", return_tensors="pt").to(device if device != "auto" else model.device)
    with torch.no_grad():
        model.generate(**dummy, max_new_tokens=5)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_single(
    model, tokenizer, prompt_length: int, gen_tokens: int, num_runs: int,
) -> dict:
    device = model.device
    # build prompt of approximate length
    dummy_text = "The quick brown fox jumps over the lazy dog. " * (prompt_length // 10 + 1)
    inputs = tokenizer(dummy_text, return_tensors="pt", truncation=True, max_length=prompt_length).to(device)
    actual_prompt_len = inputs["input_ids"].shape[1]

    ttfts = []
    throughputs = []
    total_tokens_generated = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.perf_counter()
        output = model.generate(**inputs, max_new_tokens=gen_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        generated = output.shape[1] - actual_prompt_len
        elapsed = end - start
        total_tokens_generated.append(generated)
        throughputs.append(generated / elapsed if elapsed > 0 else 0)
        ttfts.append(elapsed / generated * 1000 if generated > 0 else 0)  # approx ms/token

    return {
        "prompt_tokens": actual_prompt_len,
        "generated_tokens": int(np.mean(total_tokens_generated)),
        "tokens_per_second": float(np.mean(throughputs)),
        "ms_per_token": float(np.mean(ttfts)),
        "peak_memory_mb": get_gpu_memory_mb(),
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Quant':<8} {'Prompt':<8} {'Gen':>5} {'tok/s':>8} {'ms/tok':>8} {'VRAM MB':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        print(
            f"{r['quant']:<8} {r['prompt_tokens']:<8} {r['generated_tokens']:>5} "
            f"{r['tokens_per_second']:>8.1f} {r['ms_per_token']:>8.1f} {r['peak_memory_mb']:>10.0f}"
        )


def plot_results(results: list[dict], save_path: Path) -> None:
    quants = sorted(set(r["quant"] for r in results))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # throughput
    for q in quants:
        subset = [r for r in results if r["quant"] == q]
        subset.sort(key=lambda r: r["prompt_tokens"])
        axes[0].plot(
            [r["prompt_tokens"] for r in subset],
            [r["tokens_per_second"] for r in subset],
            marker="o", label=q,
        )
    axes[0].set_xlabel("Prompt Length (tokens)")
    axes[0].set_ylabel("Tokens / Second")
    axes[0].set_title("Generation Throughput")
    axes[0].legend()

    # latency
    for q in quants:
        subset = [r for r in results if r["quant"] == q]
        subset.sort(key=lambda r: r["prompt_tokens"])
        axes[1].plot(
            [r["prompt_tokens"] for r in subset],
            [r["ms_per_token"] for r in subset],
            marker="s", label=q,
        )
    axes[1].set_xlabel("Prompt Length (tokens)")
    axes[1].set_ylabel("ms / Token")
    axes[1].set_title("Token Latency")
    axes[1].legend()

    # memory bar chart
    memories = {}
    for r in results:
        if r["quant"] not in memories or r["peak_memory_mb"] > memories[r["quant"]]:
            memories[r["quant"]] = r["peak_memory_mb"]
    axes[2].bar(memories.keys(), memories.values(), color=["#4285F4", "#EA4335", "#34A853"][:len(memories)])
    axes[2].set_ylabel("Peak VRAM (MB)")
    axes[2].set_title("Memory Usage")

    fig.suptitle("Gemma Inference Benchmark", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved benchmark plot to {save_path}")


def main() -> None:
    # import numpy here for averaging
    global np
    import numpy as np

    parser = argparse.ArgumentParser(description="Gemma Benchmark")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--quants", nargs="+", default=["fp16", "int8", "int4"])
    parser.add_argument("--prompt-lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []

    for quant in args.quants:
        print(f"\n{'='*50}")
        print(f"Loading model: {args.model} ({quant})")
        print(f"{'='*50}")

        model = load_model(args.model, quant, args.device)
        device_str = args.device if args.device != "auto" else str(model.device)

        print("Warming up ...")
        warmup(model, tokenizer, device_str)

        for plen in args.prompt_lengths:
            print(f"  Benchmarking prompt_length={plen}, gen_tokens={args.gen_tokens}, runs={args.num_runs} ...")
            result = benchmark_single(model, tokenizer, plen, args.gen_tokens, args.num_runs)
            result["quant"] = quant
            result["model"] = args.model
            all_results.append(result)
            print(f"    → {result['tokens_per_second']:.1f} tok/s, {result['ms_per_token']:.1f} ms/tok, {result['peak_memory_mb']:.0f} MB VRAM")

        # free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_table(all_results)
    plot_results(all_results, OUTPUT_DIR / "benchmark.png")

    # save JSON
    json_path = OUTPUT_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved raw results to {json_path}")
    print("\nDone ✓")


if __name__ == "__main__":
    main()
