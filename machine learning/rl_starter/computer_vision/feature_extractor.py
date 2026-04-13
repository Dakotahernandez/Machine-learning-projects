"""
feature_extractor.py — Extract deep features and cluster/retrieve images.

Uses frozen pretrained CNNs to extract feature vectors,
then clusters them with KMeans and visualises with UMAP.

Usage:
    python feature_extractor.py --dataset cifar100 --n-clusters 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.cluster import KMeans
from tqdm import tqdm

OUTPUT_DIR = Path("outputs")


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_extractor(model_name: str = "resnet18") -> tuple[nn.Module, int]:
    """Build a feature extractor by removing the classification head."""
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dim = 512
        model.fc = nn.Identity()
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        dim = 2048
        model.fc = nn.Identity()
    else:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        dim = 512
        model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, dim


@torch.no_grad()
def extract_features(model, loader, device, max_samples: int = 5000):
    features_list, labels_list = [], []
    count = 0
    for images, labels in tqdm(loader, desc="Extracting features"):
        feats = model(images.to(device)).cpu().numpy()
        features_list.append(feats)
        labels_list.append(labels.numpy())
        count += feats.shape[0]
        if count >= max_samples:
            break
    features = np.concatenate(features_list)[:max_samples]
    labels = np.concatenate(labels_list)[:max_samples]
    return features, labels


def save_umap(features: np.ndarray, labels: np.ndarray, cluster_labels: np.ndarray, tag: str) -> None:
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping UMAP plot")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    print("  Computing UMAP projection ...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    coords = reducer.fit_transform(features)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab20", s=3, alpha=0.5)
    ax1.set_title("UMAP — True Labels")
    fig.colorbar(scatter1, ax=ax1)

    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=cluster_labels, cmap="tab20", s=3, alpha=0.5)
    ax2.set_title("UMAP — KMeans Clusters")
    fig.colorbar(scatter2, ax=ax2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{tag}_umap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {OUTPUT_DIR / f'{tag}_umap.png'}")


def find_nearest(features: np.ndarray, query_idx: int, top_k: int = 5) -> list[int]:
    query = features[query_idx]
    dists = np.linalg.norm(features - query, axis=1)
    return np.argsort(dists)[1 : top_k + 1].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature Extraction + Clustering")
    parser.add_argument("--dataset", choices=["cifar100"], default="cifar100")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--n-clusters", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    tag = f"features_{args.model}_{args.dataset}"

    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.dataset == "cifar100":
        ds = datasets.CIFAR100("data", train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    model, feat_dim = build_extractor(args.model)
    model = model.to(device)

    print(f"Extracting {args.model} features (dim={feat_dim}) ...")
    features, labels = extract_features(model, loader, device, args.max_samples)
    print(f"Features shape: {features.shape}")

    # cluster
    print(f"Clustering with KMeans (k={args.n_clusters}) ...")
    km = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(features)

    # evaluate clustering
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)
    print(f"  ARI: {ari:.4f}  NMI: {nmi:.4f}")

    # nearest neighbours demo
    print("\nNearest-Neighbour Retrieval (sample):")
    for qi in [0, 100, 500]:
        neighbours = find_nearest(features, qi, top_k=5)
        true_label = labels[qi]
        neighbour_labels = [labels[n] for n in neighbours]
        print(f"  Query {qi} (class {true_label}) → neighbours: {neighbour_labels}")

    save_umap(features, labels, cluster_labels, tag)

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(OUTPUT_DIR / f"{tag}.npz", features=features, labels=labels, clusters=cluster_labels)
    print("Done ✓")


if __name__ == "__main__":
    main()
