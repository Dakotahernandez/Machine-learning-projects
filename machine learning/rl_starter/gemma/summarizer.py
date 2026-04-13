"""
summarizer.py — Document summarisation with Gemma.

Summarises text of arbitrary length by chunking the input
and optionally performing hierarchical (recursive) summarisation.

Features:
  - Single-pass summarisation for short texts
  - Chunked summarisation for texts exceeding the context window
  - Hierarchical recursive merging of chunk summaries
  - Bullet-point or prose output styles
  - Adjustable compression ratio via max output tokens
  - Quantisation support (INT8 / INT4)
  - Reads from file, stdin, or built-in demo text

Usage:
    python summarizer.py --file paper.txt --style bullets
    python summarizer.py --text "Long article ..." --max-tokens 200
    python summarizer.py --file paper.txt --quant int4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_DIR = Path("outputs")

DEMO_TEXT = """
Artificial intelligence has undergone a remarkable transformation over the past decade,
driven primarily by advances in deep learning and the availability of massive datasets.
The introduction of the transformer architecture in 2017 marked a turning point, enabling
models to process sequences in parallel and capture long-range dependencies far more
effectively than recurrent neural networks. This led to an explosion of large language
models — from BERT and GPT-2 to GPT-4, Gemini, and open models like LLaMA and Gemma.

These models have demonstrated capabilities that were previously thought to require
human-level understanding: coherent text generation, complex reasoning, code synthesis,
and multilingual translation. However, they also introduce challenges around
hallucination, bias, safety, and the enormous computational resources required for
training. Techniques such as reinforcement learning from human feedback (RLHF),
constitutional AI, and retrieval-augmented generation (RAG) have emerged to mitigate
some of these issues.

The open-source movement has been particularly significant. Models released under
permissive licenses allow researchers and developers to fine-tune, inspect, and deploy
LLMs without relying solely on proprietary APIs. Google's Gemma family, for instance,
provides state-of-the-art performance at accessible model sizes (2B, 9B, 27B parameters),
enabling local experimentation on consumer hardware when combined with quantisation
techniques like GPTQ and bitsandbytes.

Fine-tuning has also evolved. Rather than updating all model parameters, parameter-
efficient methods like LoRA and QLoRA inject small trainable matrices into the frozen
base model, reducing memory and compute requirements by orders of magnitude. This
democratises customisation: a single GPU can fine-tune a 7B-parameter model in hours
rather than days.

Looking ahead, the field is moving towards multimodal models that combine text, images,
audio, and video understanding. Mixture-of-experts architectures promise better scaling
by activating only a subset of parameters per input. And efficient inference techniques —
speculative decoding, continuous batching, KV-cache optimisation — are making deployment
more practical for real-world applications.
""".strip()


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

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def chunk_text(text: str, tokenizer, max_chunk_tokens: int = 1500) -> list[str]:
    """Split text into chunks that fit within the token budget."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], []
    current_len = 0

    for sent in sentences:
        sent_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        if current_len + sent_tokens > max_chunk_tokens and current:
            chunks.append(". ".join(current) + ".")
            current, current_len = [], 0
        current.append(sent.strip())
        current_len += sent_tokens

    if current:
        chunks.append(". ".join(current) + ".")
    return chunks


@torch.no_grad()
def summarise_chunk(text: str, model, tokenizer, style: str, max_tokens: int) -> str:
    style_instruction = (
        "Summarise the following text as a concise bullet-point list."
        if style == "bullets"
        else "Summarise the following text in a concise paragraph."
    )
    prompt = (
        f"<start_of_turn>user\n{style_instruction}\n\n{text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def summarise(text: str, model, tokenizer, style: str, max_tokens: int) -> str:
    """Summarise text, chunking and merging recursively if needed."""
    token_count = len(tokenizer.encode(text, add_special_tokens=False))

    # fits in one pass
    if token_count <= 1500:
        print("Single-pass summarisation ...")
        return summarise_chunk(text, model, tokenizer, style, max_tokens)

    # chunk → summarise each → merge
    chunks = chunk_text(text, tokenizer, max_chunk_tokens=1500)
    print(f"Text is {token_count} tokens, split into {len(chunks)} chunks")

    summaries = []
    for i, chunk in enumerate(chunks):
        print(f"  Summarising chunk {i + 1}/{len(chunks)} ...")
        s = summarise_chunk(chunk, model, tokenizer, style, max_tokens=200)
        summaries.append(s)

    merged = "\n\n".join(summaries)
    merged_tokens = len(tokenizer.encode(merged, add_special_tokens=False))

    # if merged summaries still too long, recurse
    if merged_tokens > 1500:
        print(f"Merged text is {merged_tokens} tokens, doing another pass ...")
        return summarise(merged, model, tokenizer, style, max_tokens)

    print("Merging chunk summaries into final summary ...")
    return summarise_chunk(
        f"Combine these partial summaries into one coherent summary:\n\n{merged}",
        model, tokenizer, style, max_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma Summariser")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--file", type=str, default=None, help="Path to .txt file to summarise")
    parser.add_argument("--text", type=str, default=None, help="Direct text input")
    parser.add_argument("--style", choices=["prose", "bullets"], default="prose")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # get input text
    if args.file and Path(args.file).exists():
        text = Path(args.file).read_text(encoding="utf-8").strip()
        print(f"Loaded {len(text)} chars from {args.file}")
    elif args.text:
        text = args.text
    else:
        if args.file:
            print(f"File {args.file} not found, using demo text")
        else:
            print("No input specified, using demo text")
        text = DEMO_TEXT

    model, tokenizer = load_model(args.model, args.quant, args.device)

    summary = summarise(text, model, tokenizer, args.style, args.max_tokens)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(summary)

    # save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "summary.txt"
    report = f"Style: {args.style}\nModel: {args.model}\n\n--- ORIGINAL ---\n{text[:500]}{'...' if len(text)>500 else ''}\n\n--- SUMMARY ---\n{summary}\n"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
