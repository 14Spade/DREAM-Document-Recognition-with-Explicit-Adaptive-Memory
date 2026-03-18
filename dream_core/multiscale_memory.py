"""
DREAM Multiscale Prototype Memory Wrapper
"""
import torch
import torch.nn as nn
from typing import Tuple

from .prototype_memory_module import CategoryNeuronsMemory

class DreamMultiscaleMemory(nn.Module):
    """
    DREAM Multiscale Prototype Memory Wrapper.
    
    Encapsulates three independent memory banks operating at different 
    spatial resolutions (16x16, 32x32, 64x64). It routes visual features, 
    retrieves structural prototypes, aligns their resolutions via 
    downsampling convolutions, and fuses them into a single 4096-dim representation.
    """
    def __init__(
            self,
            ema: float = 1e-5,
            ):
        super().__init__()
        
        # 1. Initialize three Prototype Memory modules at different scales
        self.memory_16 = CategoryNeuronsMemory(memory_size=2048, feature_dim=1024, ema=ema)
        self.memory_32 = CategoryNeuronsMemory(memory_size=1024, feature_dim=512, ema=ema)
        self.memory_64 = CategoryNeuronsMemory(memory_size=512, feature_dim=256, ema=ema)
        
        # 2. Cross-scale alignment networks (Downsampling Convolutions)
        # Note: Variable names strictly match GOTQwenModel (net_2, net_3) for checkpoint compatibility
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False)

    def start_step(self):
        """
        Refresh the memory snapshots across all scales.
        """
        self.memory_16.start_step()
        self.memory_32.start_step()
        self.memory_64.start_step()

    def forward(
        self, 
        feat_16: torch.Tensor,  # Shape: (B, 256, 1024) -> 16x16 flattened
        feat_32: torch.Tensor,  # Shape: (B, 1024, 512) -> 32x32 flattened
        feat_64: torch.Tensor,  # Shape: (B, 4096, 256) -> 64x64 flattened
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # --- 1. Read Operation ---
        retrieved_16, loss_16 = self.memory_16.read(query=feat_16, return_loss=True)
        retrieved_32, loss_32 = self.memory_32.read(query=feat_32, return_loss=True)
        retrieved_64, loss_64 = self.memory_64.read(query=feat_64, return_loss=True)
        
        total_sparse_loss = loss_16 + loss_32 + loss_64

        # --- 2. Spatial Restoration & Downsampling Alignment ---
        # Get batch size from input
        B = retrieved_32.size(0)
        
        # Align 32x32 -> 16x16
        h32 = w32 = int(retrieved_32.size(1) ** 0.5)
        ret_32_spatial = retrieved_32.view(B, h32, w32, -1).permute(0, 3, 1, 2) # (B, 512, 32, 32)
        aligned_32 = self.net_3(ret_32_spatial)                                 # (B, 1024, 16, 16)
        aligned_32_flat = aligned_32.flatten(2).permute(0, 2, 1)                # (B, 256, 1024)

        # Align 64x64 -> 32x32 -> 16x16
        h64 = w64 = int(retrieved_64.size(1) ** 0.5)
        ret_64_spatial = retrieved_64.view(B, h64, w64, -1).permute(0, 3, 1, 2) # (B, 256, 64, 64)
        aligned_64_step1 = self.net_2(ret_64_spatial)                           # (B, 512, 32, 32)
        aligned_64 = self.net_3(aligned_64_step1)                               # (B, 1024, 16, 16)
        aligned_64_flat = aligned_64.flatten(2).permute(0, 2, 1)                # (B, 256, 1024)

        # --- 3. Feature Fusion (Concatenation) ---
        # Structure strictly matches your target: 
        # [cnn_feature, retrieved_memory_16, aligned_32, aligned_64] -> (B, 256, 4096)
        fused_features = torch.cat([feat_16, retrieved_16, aligned_32_flat, aligned_64_flat], dim=-1)

        # --- 4. Write Operation (Consolidation) ---
        if is_training:
            with torch.set_grad_enabled(False):

                self.memory_16.write(query=feat_16, value=feat_16)
                self.memory_32.write(query=feat_32, value=feat_32)
                self.memory_64.write(query=feat_64, value=feat_64)
                
        return fused_features, total_sparse_loss