# DREAM: Document Recognition with Explicit Adaptive Memory (CVPR 2026)

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](#) [![Dataset](https://img.shields.io/badge/Dataset-DreamDoc-blue.svg)](#) This repository contains the official dataset and codebase for our CVPR 2026 paper **DREAM**. 

## 📖 About DREAM

Large multimodal models (LMMs) have shown great promise in document recognition, but they typically rely on implicit modeling where knowledge is entangled within network weights. This makes them struggle with complex, unseen layouts and hard to update. 

To address this, we propose **DREAM**, which augments document recognition models with an **explicit, adaptive, and multiscale prototype memory**. 

### Key Features:
* **Memory Retrieval & Consolidation:** Local document regions sparsely attend to a few prototypes (e.g., image borders, tilted text) to retrieve explicit structural context. The memory is continuously updated via an EMA strategy during training.
**Hierarchical Multiscale Design:** Independent prototype memory banks operate across different spatial resolutions to capture global layouts, mid-level components, and fine-grained styles[cite: 51, 180, 183].
* **Plug-and-Play Module:** DREAM can be seamlessly integrated into various encoder-decoder architectures. [cite_start]We validated it on both large multimodal document recognition (using Qwen 0.5B decoder) and handwriting text line recognition tasks.
* **Sparsity Regularization:** We introduce an entropy-based loss to encourage sparse and non-collapsing prototype assignments.


## 🗂️ DreamDoc Dataset

To thoroughly evaluate document recognition models on complex layouts and diverse stylistic patterns, we introduce **DreamDoc**, a comprehensive bilingual (Chinese-English) dataset constructed specifically for this work.

### ✨ Dataset Highlights
**Scale & Diversity**: Contains 4,908 high-quality document images (4,800 for training, 108 for testing) sourced from real-world scenarios. The dataset covers 7 distinct categories: listed-company announcements, handwritten notes, slide pages, primary/secondary school textbooks, university textbooks, magazines, and newspapers.
**Complexity**: Features challenging layouts including single-column, double-column, complex image-text mixed pages, and diverse font styles, which heavily rely on explicit structural context.
**High-Quality Transcripts**: Annotations are generated via a semi-automatic pipeline. Initial OCR results from Chandra are manually verified and corrected by trained annotators to ensure content integrity, correct reading order, and accurate handling of dense typesetting and complex layouts.

### 📥 Download Links
You can download the full DreamDoc dataset from the following platforms:

* ☁️ **[Google Drive](#)**

### 📂 Data Structure
After downloading and extracting the dataset, you will find the images organized by category, with all text annotations consolidated in a single JSON file. The directory structure is as follows:

```text
DreamDoc/
├── announce_plus/      # Additional announcements
├── announcement/       # Listed-company announcements
├── jiaocai/            # University textbooks
├── newspaper/          # Newspapers
├── notes/              # Handwritten notes
├── ppt/                # Slide pages
├── times/              # Magazines
├── yiwujiaoyu/         # Primary/secondary school textbooks
├── addnoise.py         # Utility script for adding noise/augmentations
└── label.json          # Ground truth transcripts and metadata for all images
  ```

## 🚀 Release Status

To facilitate future research, we are releasing the comprehensive **DreamDoc** dataset and the associated project codebase. The repository is currently being updated to ensure full reproducibility and transparency.

- [√] DreamDoc Dataset
- [ ] Model Architecture Definitions (`DREAM` module)
- [ ] Inference Scripts
- [ ] Training Scripts
