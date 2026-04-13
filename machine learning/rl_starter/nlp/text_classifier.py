"""
text_classifier.py — TextCNN for AG News multi-class classification.

Implements the TextCNN architecture (Kim, 2014) with parallel
convolutional filters of different widths over word embeddings.

Usage:
    python text_classifier.py --epochs 10 --device auto
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
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")
AG_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


def download_ag_news(data_dir: Path) -> tuple[list[str], list[int], list[str], list[int]]:
    """Download AG News via torchtext or fallback CSV."""
    import csv
    import urllib.request

    ag_dir = data_dir / "ag_news"
    ag_dir.mkdir(parents=True, exist_ok=True)

    base = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv"
    for name in ["train.csv", "test.csv"]:
        dest = ag_dir / name
        if not dest.exists():
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(f"{base}/{name}", dest)

    def read_csv(path):
        texts, labels = [], []
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                labels.append(int(row[0]) - 1)  # 1-indexed → 0-indexed
                texts.append(row[1] + " " + row[2])
        return texts, labels

    train_texts, train_labels = read_csv(ag_dir / "train.csv")
    test_texts, test_labels = read_csv(ag_dir / "test.csv")
    return train_texts, train_labels, test_texts, test_labels


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


class Vocabulary:
    def __init__(self, min_freq: int = 3, max_size: int = 30_000) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []

    def build(self, texts: list[str]) -> None:
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(tokenize(t))
        self.itos = ["<pad>", "<unk>"]
        for word, freq in counter.most_common(self.max_size):
            if freq < self.min_freq:
                break
            self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text: str, max_len: int = 200) -> list[int]:
        unk = self.stoi["<unk>"]
        return [self.stoi.get(w, unk) for w in tokenize(text)[:max_len]]

    def __len__(self) -> int:
        return len(self.itos)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.data = [(vocab.encode(t, max_len), l) for t, l in zip(texts, labels)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), label


def collate(batch):
    seqs, labels = zip(*batch)
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, torch.tensor(labels, dtype=torch.long)


class TextCNN(nn.Module):
    """TextCNN with multiple parallel filter widths."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_classes: int = 4,
        filter_sizes: tuple[int, ...] = (2, 3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        conv_outs = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        pooled = torch.cat(conv_outs, dim=1)
        return self.fc(self.dropout(pooled))


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(seqs)
        loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += seqs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        out = model(seqs)
        correct += (out.argmax(1) == labels).sum().item()
        total += seqs.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="AG News TextCNN")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    train_texts, train_labels, test_texts, test_labels = download_ag_news(Path("data"))
    print(f"Train: {len(train_texts):,}  Test: {len(test_texts):,}")

    vocab = Vocabulary()
    vocab.build(train_texts)
    print(f"Vocabulary: {len(vocab):,} words")

    train_ds = TextDataset(train_texts, train_labels, vocab)
    test_ds = TextDataset(test_texts, test_labels, vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=2)

    model = TextCNN(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TextCNN: {total_params:,} parameters\n")

    history = {"train_acc": [], "test_acc": []}

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        tl, ta = train_epoch(model, train_loader, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        history["train_acc"].append(ta)
        history["test_acc"].append(test_acc)
        print(f"  Epoch {epoch}  loss={tl:.4f}  train_acc={ta:.4f}  test_acc={test_acc:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_acc"], label="Train")
    ax.plot(history["test_acc"], label="Test")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.legend()
    ax.set_title("AG News TextCNN")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "textcnn_curves.png", dpi=150)
    plt.close(fig)

    torch.save(model.state_dict(), OUTPUT_DIR / "textcnn_agnews.pt")
    print(f"\nFinal test accuracy: {history['test_acc'][-1]:.4f}")
    print("Done ✓")


if __name__ == "__main__":
    main()
