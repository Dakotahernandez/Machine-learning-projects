"""
chat.py — Interactive multi-turn chat with Gemma instruction-tuned models.

Features:
  - Multi-turn conversation with full history
  - System prompt support
  - Streaming token output
  - Configurable temperature / top-p
  - Conversation export to file

Usage:
    python chat.py
    python chat.py --model google/gemma-2-9b-it --system "You are a helpful coding assistant"
    python chat.py --quantize int4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

OUTPUT_DIR = Path("outputs")


def load_model(model_name: str, quantize: str | None):
    print(f"Loading {model_name} ...")
    kwargs: dict = {"device_map": "auto", "torch_dtype": torch.float16}

    if quantize == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs.pop("torch_dtype", None)
    elif quantize == "int4":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def format_chat(messages: list[dict], tokenizer) -> str:
    """Format messages for Gemma-IT chat template."""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # fallback manual formatting
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "model" or role == "assistant":
                parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        parts.append("<start_of_turn>model\n")
        return "\n".join(parts)


def generate_response(
    model, tokenizer, messages: list[dict],
    max_tokens: int, temperature: float, top_p: float, stream: bool,
) -> str:
    prompt = format_chat(messages, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
        repetition_penalty=1.15,
    )

    if stream:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def save_conversation(messages: list[dict], model_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"chat_{ts}.json"
    data = {"model": model_name, "messages": messages}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nConversation saved to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma Chat")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--system", type=str, default=None, help="System prompt")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--quantize", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    quant = args.quantize if args.quantize != "none" else None
    model, tokenizer = load_model(args.model, quant)

    messages: list[dict] = []
    if args.system:
        messages.append({"role": "user", "content": args.system})
        messages.append({"role": "model", "content": "Understood. I'll follow those instructions."})

    print(f"\nGemma Chat ({args.model})")
    print("Type 'quit' to exit, 'save' to export conversation, 'clear' to reset\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "save":
            save_conversation(messages, args.model)
            continue
        if user_input.lower() == "clear":
            messages.clear()
            if args.system:
                messages.append({"role": "user", "content": args.system})
                messages.append({"role": "model", "content": "Understood. I'll follow those instructions."})
            print("(conversation cleared)")
            continue

        messages.append({"role": "user", "content": user_input})

        print("\nGemma: ", end="", flush=True)
        response = generate_response(
            model, tokenizer, messages,
            args.max_tokens, args.temperature, args.top_p,
            stream=not args.no_stream,
        )
        if args.no_stream:
            print(response)
        print()

        messages.append({"role": "model", "content": response})

    save_conversation(messages, args.model)
    print("Done ✓")


if __name__ == "__main__":
    main()
