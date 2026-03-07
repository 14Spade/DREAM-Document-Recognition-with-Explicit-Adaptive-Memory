# DREAM: Document Recognition with Explicit Adaptive Memory (CVPR 2026)

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](#) [![Dataset](https://img.shields.io/badge/Dataset-DreamDoc-blue.svg)](#) This repository contains the official dataset and codebase for our CVPR 2026 paper **DREAM**. 

## 📖 About DREAM

Large multimodal models (LMMs) have shown great promise in document recognition, but they typically rely on implicit modeling where knowledge is entangled within network weights. This makes them struggle with complex, unseen layouts and hard to update. 

[cite_start]To address this, we propose **DREAM**, which augments document recognition models with an **explicit, adaptive, and multiscale prototype memory**[cite: 11, 44]. 

### Key Features:
* **Memory Retrieval & Consolidation:** Local document regions sparsely attend to a few prototypes (e.g., image borders, tilted text) to retrieve explicit structural context. [cite_start]The memory is continuously updated via an EMA strategy during training[cite: 12, 13, 14].
* [cite_start]**Hierarchical Multiscale Design:** Independent prototype memory banks operate across different spatial resolutions to capture global layouts, mid-level components, and fine-grained styles[cite: 51, 180, 183].
* **Plug-and-Play Module:** DREAM can be seamlessly integrated into various encoder-decoder architectures. [cite_start]We validated it on both large multimodal document recognition (using Qwen 0.5B decoder) and handwriting text line recognition tasks[cite: 56, 57, 217].
* [cite_start]**Sparsity Regularization:** We introduce an entropy-based loss to encourage sparse and non-collapsing prototype assignments[cite: 62].

## 🚀 Release Status

To facilitate future research, we are releasing the comprehensive **DreamDoc** dataset and the associated project codebase. The repository is currently being updated to ensure full reproducibility and transparency.

- [√] DreamDoc Dataset
- [ ] Model Architecture Definitions (`DREAM` module)
- [ ] Inference Scripts
- [ ] Training Scripts
