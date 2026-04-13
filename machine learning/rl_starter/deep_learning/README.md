# Deep Learning — PyTorch Neural Networks

From-scratch PyTorch deep learning: custom CNN architectures, training loops, learning rate scheduling, and live training visualization.

## Projects

### `mnist_cnn.py` — Handwritten Digit Recognition
- Custom CNN architecture (Conv → BatchNorm → ReLU → Pool)
- OneCycleLR scheduler for fast convergence
- Achieves **99%+** test accuracy in ~5 epochs
- Saves training curves and sample predictions

### `cifar10_resnet.py` — CIFAR-10 Image Classification
- Custom ResNet-style architecture with residual blocks
- Data augmentation (random crop, horizontal flip, color jitter)
- Cosine annealing LR schedule
- Mixed precision training (AMP) for GPU speedup
- Targets **92%+** test accuracy

### `fashion_autoencoder.py` — Fashion-MNIST Autoencoder
- Convolutional autoencoder for image compression & reconstruction
- Latent space visualization with t-SNE
- Anomaly detection using reconstruction error

## Quick Start

```powershell
pip install -r requirements.txt

# MNIST CNN
python mnist_cnn.py --epochs 10 --device auto

# CIFAR-10 ResNet
python cifar10_resnet.py --epochs 50 --device auto

# Fashion Autoencoder
python fashion_autoencoder.py --epochs 20 --device auto
```
