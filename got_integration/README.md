# Integrating DREAM with GOT-OCR 2.0 (DREAM 模块集成指南)

This directory provides a practical template of how to seamlessly integrate the **DREAM Core Module** into a state-of-the-art Large Multimodal Model for document understanding. 
*(本文件夹提供了一个工程模板，展示如何将 DREAM 核心模块“即插即用”地集成到 GOT-OCR 2.0 中。)*

> **⚠️ Important Note (说明):**
> The Python files provided in this directory are **structural templates (pseudocode)**. They highlight the exact injection points for our multiscale memory module. To run the full model, please apply these structural changes to the original GOT codebase as guided below.
> *(本文件夹中的 `.py` 文件为结构化伪代码模板，旨在清晰展示 DREAM 模块的植入位置，省略了部分冗长的基础构建代码。如需完整运行，请参考以下部署流程覆盖原仓库文件。)*

---

## 📂 Directory Structure (改造前后的文件夹结构图)

To clearly illustrate how our code is injected, below is the complete directory structure of the official GOT repository after being modified with the DREAM module:
*(为了清晰说明如何植入我们的代码，以下展示了克隆 GOT 官方仓库后，经过 DREAM 模块改造的完整目录层级结构：)*

```text
GOT-OCR-2.0-master/                 <-- Clone of the official GOT repository
├── dream_module/                   <-- [🌟 NEW] Our DREAM core module copied here
│   ├── __init__.py
│   ├── loss.py
│   ├── memory_module.py
│   └── multiscale_memory.py        <-- Multiscale wrapper
│
├── GOT/                            <-- Original GOT source code
│   ├── data/
│   ├── model/
│   │   ├── GOT_ocr_2_0.py          <-- [🔥 MODIFIED] LLM backbone (Ref: got_qwen_dream.py)
│   │   ├── vision_encoder/
│   │   │   ├── vary_b.py           <-- [🔥 MODIFIED] Vision Encoder (Ref: vary_b_multi.py)
│   │   │   └── ...
│   │   └── ...
│   ├── train/                      <-- Original training scripts (No structure changes needed)
│   │   ├── train_GOT.py
│   │   └── ...
│   └── utils/
├── README.md
└── ...
```

---

## 🔗 Acknowledgement & Citation (致谢与引用)

This integration builds upon the exceptional work of the GOT-OCR 2.0 team. 
*(本集成代码基于 GOT-OCR 2.0 团队的杰出工作。建议访问其原始仓库获取完整流程：)*
* **Original Repository**: [Ucas-HaoranWei/GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)

If you use this integrated codebase, please consider citing both our DREAM paper and the original GOT paper:
*(如果您在研究中使用了本集成代码，请考虑同时引用我们的 DREAM 论文与 GOT 原始论文：)*

```bibtex
% Citation for GOT-OCR 2.0
@article{wei2024general,
  title={General OCR Theory: Towards OCR-2.0 via a Unified End-to-End Model},
  author={Wei, Haoran and Liu, Chenglong and Chen, Jinyue and Wang, Jia and Kong, Lingyu and Xu, Yanming and Ge, Zheng and Zhao, Liang and Sun, Jianjian and Peng, Yuang and others},
  journal={arXiv preprint arXiv:2409.01704},
  year={2024}
}
```

---

## 🛠️ Deployment Pipeline (完整部署实操指南)

### Step 1: Clone the Original Repository (克隆原始仓库)
Get the official GOT-OCR 2.0 codebase:
*(获取官方的 GOT-OCR 2.0 代码：)*
```bash
git clone https://github.com/Ucas-HaoranWei/GOT-OCR2.0.git
cd GOT-OCR2.0/GOT-OCR-2.0-master
```

### Step 2: Inject the DREAM Core Module (植入 DREAM 核心模块)
Copy the `dream_module` folder provided in our repository into the root of the GOT source code (at the same level as the `GOT/` folder).
*(将本仓库提供的 `dream_module` 文件夹整体复制到 GOT 源码的根目录下：)*
```bash
cp -r /path/to/DREAM/dream_module ./
```

### Step 3: Modify the Vision Encoder (修改视觉编码器)
The original GOT vision tower outputs a single-scale feature map. Open the original `GOT/model/vision_encoder/vary_b.py`. Cross-referencing our `vary_b_multi.py` template, you need to:
1. Inject the downsampling convolutions (`net_2` and `net_3`) into the `__init__` function of `ImageEncoderViT`.
2. Modify the return logic in the `forward` function to output 3 hierarchical scales (`x3, x2, x1`).

*(原始 GOT 视觉编码器仅输出单尺度特征。请打开原始的 `vary_b.py`，对照我们提供的 `vary_b_multi.py` 模板，在 `ImageEncoderViT` 类中加入降采样卷积，并修改 `forward` 返回 3 个尺度的特征。)*

### Step 4: Modify the LLM Backbone (修改 LLM 主干与记忆特征融合)
Open the original `GOT/model/GOT_ocr_2_0.py`. Find the sections highlighted by our `[DREAM INJECTION]` tags in the `got_qwen_dream.py` template:
1. **Initialization**: Instantiate `DreamMultiscaleMemory` inside `GOTQwenModel` and expand the projector `mm_projector_memory` to 4096 dimensions.
2. **Fusion**: Inside the image feature processing loop, call `self.dream_memory(...)` to read, align, and fuse the multiscale features.
3. **Loss Function**: Add the returned sparsity regularization loss to the cross-entropy loss in `GOTQwenForCausalLM`.

*(请打开原始的 `GOT_ocr_2_0.py`，对照 `got_qwen_dream.py` 模板：实例化多尺度记忆封装器、在特征处理循环中调用它完成拼接，并在分类损失中加入稀疏正则化损失。)*

### Step 5: Configure Loss Weight and Train (配置损失权重并开始训练)
Ensure that the sparsity regularization loss weight is configured in your hyperparameters before training:
*(在启动训练前，请确保配置了稀疏正则化损失系数：)*
```python
loss_sparse_weight = 0.1  # Default coefficient used in our CVPR 2026 paper
```
You can now run the standard GOT-OCR 2.0 training scripts (e.g., `train_GOT.py`). The FP32 computational island and the EMA momentum updates will be handled automatically in the background.
*(现在您可以直接运行标准的训练脚本，将自动激活混合精度隔离与原型记忆的 EMA 动量更新。)*

## 🚀 Upgrade to Qwen3-0.6B (进阶：升级至 Qwen3)

If you wish to use the latest **Qwen3-0.6B** as the LLM backbone, please follow the specialized template provided in `got_qwen3_0.6B_dream.py`.
*(如果您希望使用最新的 **Qwen3-0.6B** 作为 LLM 底座，请参考 `got_qwen3_0.6B_dream.py` 提供的专用模板。)*

### 1. Implementation Guide (实现指南)
- **Download Weights**: Use `snapshot_download(repo_id="Qwen/Qwen3-0.6B", local_dir="./Qwen3-0.6B")` via Hugging Face Hub.
- **Class Update**: Change the parent class of `GOTQwenModel` to `Qwen3Model` and `GOTQwenForCausalLM` to `Qwen3ForCausalLM`.
- **Dimension Alignment**: Ensure `mm_projector_memory` maps to the new `config.hidden_size` (Qwen3 may differ from Qwen2).

> **⚠️ Performance Note (性能提示):**
> According to our experimental results, upgrading the backbone to **Qwen3-0.6B does not provide any performance gain on the Fox dataset**. We recommend using the default Qwen2 setup for standard reproduction of our paper results.
> *(根据我们的实验结果，将底座升级到 **Qwen3-0.6B 并不会在 Fox 数据集上带来任何性能提升**。为了标准复现论文结果，建议沿用默认的 Qwen2 配置。)*