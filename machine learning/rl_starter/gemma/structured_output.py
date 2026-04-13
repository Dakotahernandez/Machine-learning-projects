"""
structured_output.py — Schema-constrained generation with Gemma.

Prompts Gemma to produce structured JSON output matching a
user-defined schema, with validation and retry.

Features:
  - Predefined schemas (person, product, event, recipe)
  - Custom schema from JSON file
  - JSON extraction with regex fallback
  - Automatic retry on validation failure (up to 3 attempts)
  - Temperature annealing across retries for reliability
  - Batch extraction from multiple inputs
  - Quantisation support (INT8 / INT4)

Usage:
    python structured_output.py --schema person --input "John Doe is a 32 year old engineer from San Francisco"
    python structured_output.py --schema recipe --input "Classic pancakes recipe with flour, eggs, milk"
    python structured_output.py --schema-file custom.json --input "Some text" --quant int4
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_DIR = Path("outputs")

SCHEMAS = {
    "person": {
        "name": "string",
        "age": "integer",
        "occupation": "string",
        "location": "string",
    },
    "product": {
        "name": "string",
        "category": "string",
        "price": "number",
        "features": "list of strings",
        "rating": "number (1-5)",
    },
    "event": {
        "title": "string",
        "date": "string (YYYY-MM-DD)",
        "location": "string",
        "description": "string",
        "attendees": "integer",
    },
    "recipe": {
        "name": "string",
        "servings": "integer",
        "prep_time_minutes": "integer",
        "ingredients": "list of strings",
        "steps": "list of strings",
    },
}

DEMO_INPUTS = {
    "person": "Marie Curie was a Polish-French physicist born in 1867 who won two Nobel Prizes. She worked at the University of Paris.",
    "product": "The UltraBook Pro 15 is a premium laptop priced at $1299. It features a 15-inch OLED display, 32 GB RAM, and all-day battery life. Users rate it 4.7 out of 5.",
    "event": "PyCon 2024 will be held on May 15-23 in Pittsburgh, Pennsylvania. It is the largest annual gathering of the Python community with over 2500 attendees.",
    "recipe": "Classic pancakes: serves 4, takes 15 minutes to prep. You need 1.5 cups flour, 3.5 tsp baking powder, 1 tbsp sugar, 1.25 cups milk, 1 egg, 3 tbsp melted butter. Mix dry ingredients, add wet, stir until just combined. Cook on griddle until bubbles form, flip, cook until golden.",
}


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


def extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from generated text."""
    # try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # find JSON block in markdown
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if md_match:
        try:
            return json.loads(md_match.group(1))
        except json.JSONDecodeError:
            pass

    # find first {...}
    brace_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def validate_schema(data: dict, schema: dict) -> tuple[bool, list[str]]:
    """Check that all schema keys are present in the output."""
    missing = [k for k in schema if k not in data]
    return len(missing) == 0, missing


@torch.no_grad()
def generate_structured(
    text: str,
    schema: dict,
    schema_name: str,
    model,
    tokenizer,
    max_tokens: int = 512,
    max_retries: int = 3,
) -> tuple[dict | None, str]:
    schema_str = json.dumps(schema, indent=2)

    for attempt in range(max_retries):
        temp = max(0.1, 0.5 - attempt * 0.15)  # anneal temperature

        prompt = (
            f"<start_of_turn>user\n"
            f"Extract structured information from the text below.\n"
            f"Output ONLY valid JSON matching this schema:\n{schema_str}\n\n"
            f"Text: {text}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temp,
            do_sample=True,
            top_p=0.9,
        )
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        data = extract_json(raw)

        if data is not None:
            valid, missing = validate_schema(data, schema)
            if valid:
                return data, raw
            print(f"  Attempt {attempt + 1}: missing fields {missing}, retrying ...")
        else:
            print(f"  Attempt {attempt + 1}: no valid JSON found, retrying ...")

    return data, raw  # return best-effort


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma Structured Output")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--schema", choices=list(SCHEMAS.keys()), default="person")
    parser.add_argument("--schema-file", type=str, default=None, help="Custom schema JSON file")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--all-demos", action="store_true", help="Run all demo schemas")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.quant, args.device)

    # schema
    if args.schema_file and Path(args.schema_file).exists():
        schema = json.loads(Path(args.schema_file).read_text(encoding="utf-8"))
        schema_name = Path(args.schema_file).stem
    else:
        schema = SCHEMAS[args.schema]
        schema_name = args.schema

    if args.all_demos:
        tasks = [(name, SCHEMAS[name], DEMO_INPUTS[name]) for name in SCHEMAS]
    else:
        text = args.input or DEMO_INPUTS.get(schema_name, DEMO_INPUTS["person"])
        if not args.input:
            print(f"No --input specified, using demo text for '{schema_name}'")
        tasks = [(schema_name, schema, text)]

    all_results = []
    for name, sch, txt in tasks:
        print(f"\n{'='*50}")
        print(f"Schema: {name}")
        print(f"Input: {txt[:100]}{'...' if len(txt)>100 else ''}")
        print(f"{'='*50}")

        data, raw = generate_structured(txt, sch, name, model, tokenizer, args.max_tokens)

        if data:
            print("\nExtracted JSON:")
            print(json.dumps(data, indent=2))
            valid, missing = validate_schema(data, sch)
            status = "PASS" if valid else f"PARTIAL (missing: {missing})"
            print(f"Validation: {status}")
        else:
            print("\nFailed to extract valid JSON.")
            print(f"Raw output:\n{raw}")

        all_results.append({"schema": name, "input": txt, "output": data, "raw": raw})

    # save
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "structured_output.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
