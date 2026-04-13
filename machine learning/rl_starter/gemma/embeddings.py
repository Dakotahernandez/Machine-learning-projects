"""
embeddings.py — Sentence embeddings and similarity with Gemma.

Extracts dense embeddings from Gemma's hidden states, then
demonstrates similarity search, clustering, and visualisation.

Features:
  - Mean-pooled last-hidden-state embeddings
  - Pairwise cosine similarity matrix
  - Semantic search (query → ranked results)
  - K-Means clustering of embeddings
  - 2D t-SNE visualisation with cluster colours
  - Quantisation support (INT8 / INT4)

Usage:
    python embeddings.py --task similarity
    python embeddings.py --task search --query "neural networks"
    python embeddings.py --task cluster --n-clusters 3
    python embeddings.py --task all --quant int4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

OUTPUT_DIR = Path("outputs")

CORPUS = [
    # AI/ML
    "Neural networks are computational models inspired by biological neurons.",
    "Gradient descent is the primary optimisation algorithm for training deep learning models.",
    "Transformers use self-attention to process sequences in parallel.",
    "Large language models generate text by predicting the next token.",
    # Programming
    "Python is a popular language for data science and machine learning.",
    "JavaScript powers interactive websites and server-side applications with Node.js.",
    "Rust provides memory safety without garbage collection through ownership rules.",
    "Git is a distributed version control system for tracking code changes.",
    # Science
    "DNA carries genetic information and is structured as a double helix.",
    "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
    "Quantum mechanics describes the behaviour of particles at atomic scales.",
    "The speed of light in a vacuum is approximately 299,792 km per second.",
    # Daily life
    "Coffee is brewed from roasted and ground beans of the Coffea plant.",
    "Regular exercise improves cardiovascular health and mental well-being.",
    "The Mediterranean diet emphasises fruits, vegetables, whole grains, and olive oil.",
    "Reading books strengthens vocabulary, critical thinking, and empathy.",
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
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        normed = torch.nn.functional.normalize(pooled, dim=-1)
        all_embeds.append(normed.cpu().numpy())
    return np.vstack(all_embeds).astype("float32")


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return embeddings @ embeddings.T


def semantic_search(query_embed: np.ndarray, corpus_embeds: np.ndarray, corpus: list[str], top_k: int = 5):
    scores = (query_embed @ corpus_embeds.T).flatten()
    top_idx = scores.argsort()[::-1][:top_k]
    return [(corpus[i], float(scores[i])) for i in top_idx]


def plot_similarity(sim: np.ndarray, labels: list[str], save_path: Path) -> None:
    short = [s[:30] + "..." if len(s) > 30 else s for s in labels]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Cosine Similarity Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved similarity heatmap to {save_path}")


def plot_clusters(embeddings: np.ndarray, labels: list[str], clusters: np.ndarray, save_path: Path) -> None:
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap="tab10", s=80, edgecolors="k", linewidths=0.5)

    for i, label in enumerate(labels):
        short = label[:35] + "..." if len(label) > 35 else label
        ax.annotate(short, (coords[i, 0], coords[i, 1]), fontsize=6, alpha=0.8,
                     xytext=(5, 5), textcoords="offset points")

    ax.set_title("t-SNE Embedding Clusters")
    ax.legend(*scatter.legend_elements(), title="Cluster")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved cluster plot to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma Embeddings")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b-it")
    parser.add_argument("--task", choices=["similarity", "search", "cluster", "all"], default="all")
    parser.add_argument("--query", type=str, default="How do neural networks learn?")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quant", choices=["none", "int8", "int4"], default="none")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, args.quant, args.device)

    print(f"Embedding {len(CORPUS)} sentences ...")
    embeddings = embed_texts(CORPUS, model, tokenizer)
    print(f"Embedding shape: {embeddings.shape}")

    tasks = [args.task] if args.task != "all" else ["similarity", "search", "cluster"]

    if "similarity" in tasks:
        print("\n--- Cosine Similarity ---")
        sim = similarity_matrix(embeddings)
        plot_similarity(sim, CORPUS, OUTPUT_DIR / "embedding_similarity.png")

        # show most / least similar pairs
        n = len(CORPUS)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((sim[i, j], i, j))
        pairs.sort(reverse=True)
        print("\nTop-5 most similar pairs:")
        for score, i, j in pairs[:5]:
            print(f"  {score:.4f}  {CORPUS[i][:50]}  <->  {CORPUS[j][:50]}")
        print("\nTop-5 least similar pairs:")
        for score, i, j in pairs[-5:]:
            print(f"  {score:.4f}  {CORPUS[i][:50]}  <->  {CORPUS[j][:50]}")

    if "search" in tasks:
        print(f"\n--- Semantic Search ---")
        print(f"Query: {args.query}")
        q_embed = embed_texts([args.query], model, tokenizer)
        results = semantic_search(q_embed, embeddings, CORPUS, args.top_k)
        for rank, (text, score) in enumerate(results, 1):
            print(f"  {rank}. ({score:.4f}) {text}")

    if "cluster" in tasks:
        print(f"\n--- K-Means Clustering (k={args.n_clusters}) ---")
        km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
        clusters = km.fit_predict(embeddings)
        for c in range(args.n_clusters):
            members = [CORPUS[i] for i in range(len(CORPUS)) if clusters[i] == c]
            print(f"\n  Cluster {c} ({len(members)} members):")
            for m in members:
                print(f"    - {m[:70]}")
        plot_clusters(embeddings, CORPUS, clusters, OUTPUT_DIR / "embedding_clusters.png")

    print("\nDone ✓")


if __name__ == "__main__":
    main()
