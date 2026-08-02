# DREAM: Document Recognition with Explicit Adaptive Memory (CVPR 2026)

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](./Zhao_DREAM_Document_Recognition_with_Explicit_Adaptive_Memory_CVPR_2026_paper.pdf) [![Dataset](https://img.shields.io/badge/Dataset-DreamDoc-blue.svg)](https://drive.google.com/drive/folders/1nA5FXrleUefUy9vM40__jypc5TwO6PBx?usp=sharing)

**Tianqi Zhao**<sup>1</sup>, **Di Wu**<sup>1</sup>, **Liangrui Peng**<sup>1</sup>, **Yifan Huang**<sup>1</sup>, **Kemeng Zhao**<sup>1</sup>, **Shuo Li**<sup>1</sup>, **Zhiyu Li**<sup>1</sup>, **Yizhu Wang**<sup>1</sup>, **Borui Jiang**<sup>2</sup>, **Yuyang Li**<sup>2</sup>

<sup>1</sup> Tsinghua University, <sup>2</sup> Huawei Noah's Ark Lab

*(Note: The codebase is currently being updated for full public release. We are progressively uploading the core modules and scripts.)*

---

## 📖 About DREAM

Large multimodal models (LMMs) have shown great promise in document recognition, but they typically rely on implicit modeling where knowledge is entangled within network weights. This makes them struggle with complex, unseen layouts and hard to update. 

To address this, we propose **DREAM**, which augments document recognition models with an **explicit, adaptive, and multiscale prototype memory**. 

### Key Features:
* **Memory Retrieval & Consolidation:** Local document regions sparsely attend to a few prototypes (e.g., image borders, tilted text) to retrieve explicit structural context. The memory is continuously updated via an EMA strategy during training.
* **Hierarchical Multiscale Design:** Independent prototype memory banks operate across different spatial resolutions to capture global layouts, mid-level components, and fine-grained styles.
* **Plug-and-Play Module:** DREAM can be seamlessly integrated into various encoder-decoder architectures. We validated it on both large multimodal document recognition (using Qwen 0.5B/3B decoders) and handwriting text line recognition tasks.
* **Sparsity Regularization:** We introduce an entropy-based loss to encourage sparse and non-collapsing prototype assignments.

---

## 💻 Codebase Structure & Integration

The repository is modularly designed to allow seamless integration into existing vision-language pipelines. 

### 1. Core Algorithm (`dream_core/`)
The standalone implementation of the DREAM memory architecture.
* `prototype_memory_module.py`: The explicit memory bank featuring FP32-isolated Read (MHA retrieval) and Write (EMA consolidation) mechanisms.
* `multiscale_memory.py`: The central dispatcher that aligns, triggers, and fuses hierarchical visual features (16x16, 32x32, 64x64).
* `loss.py`: The sparsity entropy loss function designed to prevent memory collapse.

### 2. Model Integration (`got_integration/`)
Templates and modified architectures for injecting DREAM into GOT-OCR 2.0 (Qwen-based).
* `vary_b_multi.py`: A heavily modified Vision Tower that extracts and outputs three hierarchical feature scales simultaneously.
* `got_qwen_dream.py`: The integration template for Qwen2, showcasing the exact `[DREAM INJECTION]` locations.
* `got_qwen3_0.6B_dream.py`: Upgrade GOT's decoder to Qwen3-0.6B.

### 3. Verification & Visualization (`examples/`)
Tools to diagnose the memory data flow and visualize adaptation dynamics.
* `test_dream_core_module.py`: A lightweight diagnostic script to verify multi-scale alignment and synchronization across all memory banks.
* `run_dream_demo.py`: A high-fidelity visualization simulation that demonstrates how memory prototypes "carve" and adapt to structural archetypes over time.

---

## 🗂️ DreamDoc Dataset

To thoroughly evaluate document recognition models on complex layouts and diverse stylistic patterns, we introduce **DreamDoc**, a comprehensive bilingual (Chinese-English) dataset constructed specifically for this work.

### ✨ Dataset Highlights
* **Scale & Diversity**: Contains 4,908 high-quality document images (4,800 for training, 108 for testing) sourced from real-world scenarios. The dataset covers 7 distinct categories: listed-company announcements, handwritten notes, slide pages, primary/secondary school textbooks, university textbooks, magazines, and newspapers.
* **Complexity**: Features challenging layouts including single-column, double-column, complex image-text mixed pages, and diverse font styles, which heavily rely on explicit structural context.
* **High-Quality Transcripts**: Annotations are generated via a semi-automatic pipeline. Initial OCR results from Chandra are manually verified and corrected by trained annotators to ensure content integrity, correct reading order, and accurate handling of dense typesetting and complex layouts.

### 📥 Download Links
You can download the full DreamDoc dataset from the following platforms:

* ☁️ **[Google Drive](https://drive.google.com/drive/folders/1nA5FXrleUefUy9vM40__jypc5TwO6PBx?usp=sharing)**

### 📂 Data Structure
After downloading and extracting the dataset, you will find the images organized by category, with all text annotations consolidated in a single JSON file. The directory structure is as follows:

```text
DreamDoc Dataset V0/
├── images/       # All document images
└── label.json    # Ground-truth transcripts and metadata
```

## 🚀 Release Status

To facilitate future research, we are releasing the comprehensive **DreamDoc** dataset and the associated project codebase. The repository is currently being updated to ensure full reproducibility and transparency.

- [x] DreamDoc Dataset
- [x] Model Architecture Definitions (dream_core)
- [x] Integration Architectures for GOT-OCR (got_integration)
- [x] Diagnostic and Visualization (examples)
- [ ] Inference Scripts and Model Weights
- [ ] Training Scripts
