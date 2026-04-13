"""
sentiment_lstm.py — Bidirectional LSTM for IMDB sentiment analysis.

Features:
  - Vocabulary built from training data with frequency threshold
  - Learned word embeddings
  - Bidirectional LSTM with dropout
  - Packed sequences for efficient variable-length processing
  - Training curves and per-class metrics

Usage:
    python sentiment_lstm.py --epochs 5 --device auto
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")

# ── dataset ─────────────────────────────────────────────────────


def download_imdb(data_dir: Path) -> Path:
    """Download IMDB dataset if not present."""
    import tarfile
    import urllib.request

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = data_dir / "aclImdb_v1.tar.gz"
    extracted = data_dir / "aclImdb"
    if extracted.exists():
        return extracted
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading IMDB dataset ...")
    urllib.request.urlretrieve(url, tar_path)
    print("Extracting ...")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(data_dir, filter="data")
    tar_path.unlink()
    return extracted


def load_imdb_split(root: Path, split: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for label_name, label_id in [("pos", 1), ("neg", 0)]:
        folder = root / split / label_name
        for f in sorted(folder.glob("*.txt")):
            texts.append(f.read_text(encoding="utf-8"))
            labels.append(label_id)
    return texts, labels


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.split()


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self, min_freq: int = 5, max_size: int = 25_000) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []

    def build(self, texts: list[str]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        self.itos = [self.PAD, self.UNK]
        for word, freq in counter.most_common(self.max_size):
            if freq < self.min_freq:
                break
            self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        unk_id = self.stoi[self.UNK]
        return [self.stoi.get(w, unk_id) for w in tokenize(text)]

    def __len__(self) -> int:
        return len(self.itos)


class IMDBDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary, max_len: int = 300) -> None:
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids = self.vocab.encode(self.texts[idx])[: self.max_len]
        return torch.tensor(ids, dtype=torch.long), self.labels[idx], len(ids)


def collate_fn(batch):
    seqs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, labels, lengths


# ── model ───────────────────────────────────────────────────────


class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True, dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        # concat final forward + backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden)).squeeze(1)


# ── training ────────────────────────────────────────────────────


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels, lengths in loader:
        seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        logits = model(seqs, lengths)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        preds = (logits > 0).long()
        correct += (preds == labels.long()).sum().item()
        total += seqs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels, lengths in loader:
        seqs, labels, lengths = seqs.to(device), labels.to(device), lengths.to(device)
        logits = model(seqs, lengths)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item() * seqs.size(0)
        preds = (logits > 0).long()
        correct += (preds == labels.long()).sum().item()
        total += seqs.size(0)
    return total_loss / total, correct / total


def save_curves(history: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
    fig.suptitle("IMDB Sentiment LSTM")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sentiment_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / 'sentiment_curves.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IMDB Sentiment LSTM")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    # data
    imdb_root = download_imdb(Path("data"))
    train_texts, train_labels = load_imdb_split(imdb_root, "train")
    test_texts, test_labels = load_imdb_split(imdb_root, "test")

    vocab = Vocabulary(min_freq=5, max_size=25_000)
    vocab.build(train_texts)
    print(f"Vocabulary: {len(vocab):,} words")

    train_ds = IMDBDataset(train_texts, train_labels, vocab)
    test_ds = IMDBDataset(test_texts, test_labels, vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = SentimentLSTM(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SentimentLSTM: {total_params:,} parameters\n")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        tl, ta = train_epoch(model, train_loader, optimizer, device)
        vl, va = evaluate(model, test_loader, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"  Epoch {epoch}  loss={tl:.4f}  acc={ta:.4f}  val_loss={vl:.4f}  val_acc={va:.4f}")

    save_curves(history)
    OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_DIR / "sentiment_lstm.pt")
    print(f"\nFinal test accuracy: {history['val_acc'][-1]:.4f}")
    print("Done ✓")


if __name__ == "__main__":
    main()
