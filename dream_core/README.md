# DREAM Core Module

This folder contains the core implementation of the **Explicit Adaptive Multiscale Memory**, as proposed in our CVPR 2026 paper: *DREAM: Document Recognition with Explicit Adaptive Memory*.

We have intentionally decoupled this module to serve as a **plug-and-play** component. It can be seamlessly integrated into any Transformer- or CNN-based visual encoder to enhance document recognition models with nonparametric structural and stylistic knowledge.

## 📂 File Structure

* `loss.py`: Implements the **Sparsity and Anti-collapse Regularization**. It enforces sharp attention distributions to prevent prototype collapse across the training corpus.
* `prototype_memory_module.py`: Implements the **Single-scale Prototype Memory (`CategoryNeuronsMemory`)**. Features a float32 computational island for mixed-precision stability and a gradient-free Exponential Moving Average (EMA) update mechanism.
* `multiscale_memory.py`: Implements the **Multiscale Memory Wrapper (`DreamMultiscaleMemory`)**. Demonstrates how to handle multi-resolution visual features (e.g., `16x16`, `32x32`, `64x64`), align them via downsampling convolutions, and fuse them for the LLM decoder.

## 🚀 Quick Start: Plug-and-Play

You can inject strong document layout priors into your own Large Multimodal Model (LMM) with just a few lines of code:

```python
import torch
from dream_module.multiscale_memory import DreamMultiscaleMemory

# 1. Instantiate the multiscale memory module
dream_memory = DreamMultiscaleMemory().cuda()

# 2. Refresh the memory snapshot before each forward pass
dream_memory.start_step()

# 3. Simulate multi-resolution visual features extracted by your Vision Encoder
feat_16 = torch.randn(2, 256, 1024).cuda()  # Coarse scale
feat_32 = torch.randn(2, 1024, 512).cuda()  # Mid scale
feat_64 = torch.randn(2, 4096, 256).cuda()  # Fine scale

# 4. Execute read, multiscale alignment, fusion, and background EMA write
fused_features, sparse_loss = dream_memory(
    feat_16, feat_32, feat_64, 
    is_training=True
)

print("Fused Features Shape:", fused_features.shape) # Expected: (2, 256, 4096)
print("Sparsity Regularization Loss:", sparse_loss.item())

# 5. Project the fused_features and feed them to your LLM Decoder!