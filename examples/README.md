# DREAM: Examples & Simulations

This directory contains diagnostic and simulation scripts to verify and visualize the behavior of the **DREAM (Document Recognition with Explicit Adaptive Memory)** module. 

---

## 🛠️ 1. `test_dream_core_module.py`
**Multi-scale Integration & Diagnostic Tool**

This script serves as a lightweight, single-pass diagnostic test to verify the structural integrity and data flow of the hierarchical memory system.

### What it does:
It simulates a single forward pass of visual features across three different scales (16x16, 32x32, and 64x64) through the DREAM module. It outputs a detailed, step-by-step terminal report verifying that:
* **Read & Align:** Features from different scales are correctly retrieved and spatially down-sampled/aligned.
* **Write (EMA):** The Exponential Moving Average (EMA) update mechanism is successfully triggered across all corresponding memory banks.
* **Projection:** The fused multi-scale features are correctly concatenated and projected back to the required Large Language Model (LLM) hidden dimension.

**Usage:**
```bash
python examples/test_dream_core_module.py
```

---

## 🚀 2. `run_dream_demo.py`
**Memory Adaptation Training Simulation**

This script is a visualization tool designed to demonstrate the long-term learning and convergence dynamics of the DREAM module. 

### The Toy Example: Simulating Structural Consistency
To illustrate how the memory module adapts to document layouts, this script employs a controlled **toy example**. Instead of feeding a diverse dataset, we repeatedly feed the **exact same fixed document sample** into the memory module for 80 iterations. 

This extreme scenario simulates a real-world situation where a model processes massive volumes of documents sharing a highly consistent structural layout (e.g., standardized invoices or forms). By repeatedly observing the same structural archetype, we can clearly visualize how the memory system uses **SGD** (to learn projections) and **EMA** (to update memory weights) to slowly "memorize" and reconstruct this layout.

### The Visualization Output
Upon completion, the script generates a 3-pane analytical plot (`dream_memory_visualize.png`) to quantify the memory adaptation process:

* **Left Plot (Input Visual Query):** Displays the feature map of the original, fixed document sample. 
    * **Y-axis:** Represents the index of the first 50 spatial tokens.
    * **X-axis:** Represents the feature dimension channels.
* **Middle Plot (Retrieved Memory):** Displays the reconstructed feature map retrieved from the memory module after the training steps. Comparing this to the left plot visually demonstrates how accurately the memory bank has learned and "carved" the input structure.
* **Right Plot (Memory Adaptation Dynamics):** A dual-axis graph tracking the optimization metrics over the training steps. 
    * The **Red Curve (Sparsity Loss)** shows the regularization term stabilizing to prevent memory collapse.
    * The **Blue Curve (Cosine Similarity)** illustrates the alignment accuracy between the input query and the retrieved memory climbing as the module learns.

**Usage:**
```bash
python examples/run_dream_demo.py
```