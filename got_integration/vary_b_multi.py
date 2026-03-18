"""
Modified Vision Encoder for DREAM: vary_b_multi.py

NOTE (HOW TO USE): 
This is a structural template. To create the fully runnable file:
1. Copy the entire content of the original `GOT/model/vision_encoder/vary_b.py`.
2. Locate the `ImageEncoderViT` class.
3. Inject our `net_2` and `net_3` definitions into the `__init__` function.
4. Replace the return logic in the `forward` function as shown below.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type
from functools import partial

# ====================================================================
# [OMITTED ORIGINAL CODE 1]
# Please KEEP the original definitions of the following classes from GOT-OCR 2.0:
# - LayerNorm2d
# - MLPBlock
# - Block
# - Attention
# - PatchEmbed
# - window_partition / window_unpartition / get_rel_pos / add_decomposed_rel_pos
# ====================================================================

class ImageEncoderViT(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # ====================================================================
        # [OMITTED ORIGINAL CODE 2]
        # Please KEEP the original initialization logic here, including:
        # self.patch_embed = PatchEmbed(...)
        # self.pos_embed = nn.Parameter(...)
        # self.blocks = nn.ModuleList(...)
        # self.neck = nn.Sequential(...)
        # ====================================================================

        # ====================================================================
        # [DREAM INJECTION]: Multiscale Feature Downsampling Layers
        # ====================================================================
        # Generate 32x32 features from 64x64
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        # Generate 16x16 features from 32x32
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x) # Output shape: [B, 64, 64, 768]

        # ====================================================================
        # [DREAM INJECTION]: Output 3 Hierarchical Scales
        # ====================================================================
        x1 = self.neck(x.permute(0, 3, 1, 2))  # Fine scale: [B, 256, 64, 64]
        x2 = self.net_2(x1)                    # Mid scale:  [B, 512, 32, 32]
        x3 = self.net_3(x2)                    # Coarse scale: [B, 1024, 16, 16]

        # Returns 3 scales to be consumed by DreamMultiscaleMemory
        return x3, x2, x1

# ====================================================================
# [OMITTED ORIGINAL CODE 3]
# Please KEEP the original wrapper functions at the bottom of the file:
# - build_vary_vit_b()
# - _build_vary()
# ====================================================================