"""
rag_pipeline.py — Retrieval-Augmented Generation with Gemma.

Embeds documents using Gemma's hidden states, stores them in a
FAISS index, retrieves the most relevant chunks, and generates
grounded answers.

Features:
  - Mean-pooled last-hidden-state embeddings
  - FAISS inner-product (cosine) index
  - Top-k retrieval with relevance scores
  - Context-aware prompt construction
  - Built-in demo corpus if no documents supplied
  - Save / load index and chunks
  - Quantisation support (INT8 / INT4)

Usage:
    python rag_pipeline.py --query "How do transformers work?"
    python rag_pipeline.py --docs documents/ --query "Explain attention"
    python rag_pipeline.py --query "What is LoRA?" --quant int4
"""

from __future__ import annotations

import argparse
import pickle
import textwrap
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_DIR = Path("outputs")

DEMO_CORPUS = [
    "Transformers are a neural network architecture that uses self-attention to process sequences in parallel. They replaced recurrent models for most NLP tasks because they are faster and capture long-range dependencies better.",
    "The attention mechanism computes a weighted sum of value vectors, where the weights are determined by the compatibility between query and key vectors. Multi-head attention runs several attention functions in parallel.",
    "LoRA (Low-Rank Adaptation) freezes the pre-trained model weights and injects trainable rank-decomposition matrices into each transformer layer, dramatically reducing the number of trainable parameters for fine-tuning.",
    "Quantisation reduces the numerical precision of model weights (e.g., from FP32 to INT8 or INT4) to shrink memory usage and speed up inference with minimal accuracy loss.",
    "Retrieval-augmented generation (RAG) combines a retriever that finds relevant documents with a generator that conditions on those documents, reducing hallucination and grounding answers in factual sources.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors. It supports exact and approximate nearest-neighbour search on billions of vectors.",
    "Gemma is an open model family by Google built on the same technology as Gemini. It comes in 2B, 9B, and 27B parameter sizes with instruction-tuned variants.",
    "Gradient checkpointing trades compute for memory by recomputing intermediate activations during the backward pass instead of storing them.",
    "The softmax function converts a vector of real numbers into a probability distribution. In attention, it normalises the compatibility scores so they sum to one.",
    "Beam search explores multiple candidate sequences simultaneously and keeps the top-B most likely sequences at each generation step, often producing more coherent text than greedy decoding.",
    "Cosine similarity measures the angle between two vectors. A value of 1 means identical direction, 0 means orthogonal, and -1 means opposite direction.",
    "Embedding models represent text as dense vectors in a continuous space where semantically similar texts have nearby representations.",
]


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


@torch.no_grad()
def embed_texts(texts: list[str], model, tokenizer, batch_size: int = 8) -> np.ndarray:
    """Mean-pool the last hidden state to get sentence embeddings."""
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # (B, T, D)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        normed = torch.nn.functional.normalize(pooled, dim=-1)
        all_embeds.append(normed.cpu().numpy())
    return np.vstack(all_embeds).astype("float32")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def retrieve(query_embed: np.ndarray, index, chunks: list[str], top_k: int = 3):
    scores, indices = index.search(query_embed, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({"chunk": chunks[idx], "score": float(score), "index": int(idx)})
    return results


def load_documents(docs_path: str | None) -> list[str]:
    if docs_path and Path(docs_path).exists():
        p = Path(docs_path)
        texts = []
        for f in sorted(p.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8").strip())
        if texts:
            print(f"Loaded {len(texts)} documents from {p}")
            return texts
    print("Using built-in demo corpus (12 passages)")
    return DEMO_CORPUS


def generate_answer(query: str, context: list[str], model, tokenizer, max_new: int = 256) -> str:
    ctx_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
    prompt = (
        f"<start_of_turn>user\nAnswer the question using only the provided context.\n\n"
        f"Context:\n{ctx_block}\n\nQuestion: {query}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma RAG Pipeline")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--docs", type=str, default=None, help="Directory of .txt files")
    parser.add_argument("--query", type=str, default="How do transformers work?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-index", action="store_true")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.quant, args.device)
    chunks = load_documents(args.docs)

    # embed
    print(f"Embedding {len(chunks)} chunks ...")
    embeddings = embed_texts(chunks, model, tokenizer)
    index = build_index(embeddings)
    print(f"Index built: {index.ntotal} vectors, dim={embeddings.shape[1]}")

    # save index
    if args.save_index:
        OUTPUT_DIR.mkdir(exist_ok=True)
        faiss.write_index(index, str(OUTPUT_DIR / "rag_index.faiss"))
        with open(OUTPUT_DIR / "rag_chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        print(f"Index and chunks saved to {OUTPUT_DIR}")

    # retrieve
    print(f"\nQuery: {args.query}")
    query_embed = embed_texts([args.query], model, tokenizer)
    results = retrieve(query_embed, index, chunks, top_k=args.top_k)

    print(f"\nTop-{args.top_k} retrieved passages:")
    for r in results:
        score_str = f"{r['score']:.4f}"
        print(f"  [{r['index']}] (score={score_str})")
        for line in textwrap.wrap(r["chunk"], width=80):
            print(f"      {line}")

    # generate
    context = [r["chunk"] for r in results]
    print("\nGenerating answer ...")
    answer = generate_answer(args.query, context, model, tokenizer)
    print(f"\nAnswer:\n{answer}")

    # save
    OUTPUT_DIR.mkdir(exist_ok=True)
    report = f"Query: {args.query}\n\nRetrieved:\n"
    for r in results:
        report += f"  [{r['index']}] score={r['score']:.4f}: {r['chunk']}\n"
    report += f"\nAnswer:\n{answer}\n"
    out_path = OUTPUT_DIR / "rag_result.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
