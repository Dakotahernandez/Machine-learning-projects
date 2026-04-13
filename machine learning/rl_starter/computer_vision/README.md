# Computer Vision

Transfer learning, fine-tuning pretrained models, and feature extraction with PyTorch and torchvision.

## Projects

### `transfer_learning.py` — Fine-Tune Any Pretrained Model
- Supports ResNet18/34/50, EfficientNet-B0, MobileNetV3
- Automatic dataset loading from torchvision (CIFAR-100, Flowers102, Food101)
- Progressive unfreezing: freeze backbone → train head → unfreeze & fine-tune
- Grad-CAM attention maps to visualise what the model focuses on
- Mixed precision training

### `feature_extractor.py` — Pretrained Feature Extraction + Clustering
- Extract deep features from images using frozen pretrained CNNs
- KMeans / DBSCAN clustering on extracted features
- Nearest-neighbour image retrieval (find similar images)
- UMAP visualisation of feature space

## Quick Start

```powershell
pip install -r requirements.txt

# Fine-tune ResNet18 on CIFAR-100
python transfer_learning.py --model resnet18 --dataset cifar100 --epochs 20

# Fine-tune EfficientNet on Flowers102
python transfer_learning.py --model efficientnet_b0 --dataset flowers102 --epochs 15

# Feature extraction + clustering
python feature_extractor.py --dataset cifar100 --n-clusters 20
```
