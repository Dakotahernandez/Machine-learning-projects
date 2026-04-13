"""
word_embeddings.py — Word2Vec Skip-Gram with negative sampling from scratch.

Trains word embeddings on a text corpus, visualises clusters
with t-SNE, and solves word analogies.

Usage:
    python word_embeddings.py --epochs 5
    python word_embeddings.py --corpus path/to/text.txt --epochs 10
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")


def get_sample_text() -> str:
    """Return a built-in sample corpus for demo purposes."""
    import urllib.request
    url = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
    cache = Path("data/sherlock.txt")
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    cache.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading sample corpus (Sherlock Holmes) ...")
    urllib.request.urlretrieve(url, cache)
    return cache.read_text(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z\s]", "", text.lower()).split()


class Vocabulary:
    def __init__(self, min_freq: int = 5, max_size: int = 10_000) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []
        self.freqs: np.ndarray = np.array([])

    def build(self, tokens: list[str]) -> None:
        counter = Counter(tokens)
        self.itos = []
        freqs = []
        for word, freq in counter.most_common(self.max_size):
            if freq < self.min_freq:
                break
            self.itos.append(word)
            freqs.append(freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.freqs = np.array(freqs, dtype=np.float64)
        self.freqs = self.freqs ** 0.75  # subsampling
        self.freqs /= self.freqs.sum()

    def __len__(self) -> int:
        return len(self.itos)


class SkipGramDataset(Dataset):
    def __init__(self, tokens: list[str], vocab: Vocabulary, window: int = 5) -> None:
        self.pairs: list[tuple[int, int]] = []
        ids = [vocab.stoi[t] for t in tokens if t in vocab.stoi]
        for i, center in enumerate(ids):
            start = max(0, i - window)
            end = min(len(ids), i + window + 1)
            for j in range(start, end):
                if j != i:
                    self.pairs.append((center, ids[j]))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 100) -> None:
        super().__init__()
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)
        nn.init.xavier_uniform_(self.center_embed.weight)
        nn.init.xavier_uniform_(self.context_embed.weight)

    def forward(self, center: torch.Tensor, context: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        c = self.center_embed(center)       # (batch, dim)
        ctx = self.context_embed(context)    # (batch, dim)
        neg_emb = self.context_embed(neg)    # (batch, k, dim)

        pos_score = torch.sum(c * ctx, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8)

        neg_score = torch.bmm(neg_emb, c.unsqueeze(2)).squeeze(2)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-8).sum(dim=1)

        return (pos_loss + neg_loss).mean()


def sample_negatives(batch_size: int, k: int, freqs: np.ndarray) -> torch.Tensor:
    neg = np.random.choice(len(freqs), size=(batch_size, k), p=freqs)
    return torch.from_numpy(neg).long()


def get_embeddings(model: SkipGram) -> np.ndarray:
    return model.center_embed.weight.detach().cpu().numpy()


def solve_analogy(vocab: Vocabulary, embeddings: np.ndarray, a: str, b: str, c: str, top_k: int = 5):
    """a is to b as c is to ?"""
    if a not in vocab.stoi or b not in vocab.stoi or c not in vocab.stoi:
        return []
    va = embeddings[vocab.stoi[a]]
    vb = embeddings[vocab.stoi[b]]
    vc = embeddings[vocab.stoi[c]]
    target = vb - va + vc
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    sims = (embeddings / norms) @ (target / (np.linalg.norm(target) + 1e-8))
    exclude = {vocab.stoi[a], vocab.stoi[b], vocab.stoi[c]}
    ranked = np.argsort(-sims)
    results = []
    for idx in ranked:
        if idx not in exclude:
            results.append((vocab.itos[idx], float(sims[idx])))
            if len(results) >= top_k:
                break
    return results


def save_tsne(vocab: Vocabulary, embeddings: np.ndarray, n_words: int = 200) -> None:
    from sklearn.manifold import TSNE

    OUTPUT_DIR.mkdir(exist_ok=True)
    E = embeddings[:n_words]
    words = vocab.itos[:n_words]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(E)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.6)
    for i, word in enumerate(words[:80]):  # label top 80
        ax.annotate(word, (coords[i, 0], coords[i, 1]), fontsize=7, alpha=0.8)
    ax.set_title(f"Word Embeddings t-SNE (top {n_words} words)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "word2vec_tsne.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'word2vec_tsne.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Word2Vec Skip-Gram")
    parser.add_argument("--corpus", type=str, default=None, help="Path to text file")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--neg-samples", type=int, default=5)
    args = parser.parse_args()

    if args.corpus:
        text = Path(args.corpus).read_text(encoding="utf-8")
    else:
        text = get_sample_text()

    tokens = tokenize(text)
    print(f"Corpus: {len(tokens):,} tokens")

    vocab = Vocabulary(min_freq=5, max_size=10_000)
    vocab.build(tokens)
    print(f"Vocabulary: {len(vocab):,} words")

    dataset = SkipGramDataset(tokens, vocab, window=5)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print(f"Training pairs: {len(dataset):,}\n")

    model = SkipGram(len(vocab), args.embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        total_loss = 0.0
        for center, context in loader:
            neg = sample_negatives(center.size(0), args.neg_samples, vocab.freqs)
            loss = model(center, context, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * center.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"  Epoch {epoch}  loss={avg_loss:.4f}")

    embeddings = get_embeddings(model)

    # analogies
    print("\nWord Analogies:")
    for a, b, c in [("king", "man", "woman"), ("good", "better", "bad")]:
        results = solve_analogy(vocab, embeddings, a, b, c, top_k=3)
        if results:
            answers = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {a} → {b}, {c} → {answers}")
        else:
            print(f"  {a} → {b}, {c} → (word not in vocabulary)")

    save_tsne(vocab, embeddings)

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.save(OUTPUT_DIR / "word2vec_embeddings.npy", embeddings)
    print("Done ✓")


if __name__ == "__main__":
    main()
