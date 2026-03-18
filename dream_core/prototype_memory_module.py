import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple

# Import the standalone sparse loss
from .loss import sparse_entropy_loss

def _nan_to_num_(tensor: torch.Tensor) -> torch.Tensor:
    """
    Utility function to handle potential NaN values.
    Maintained for stability within the fp32 computational island.
    """
    return tensor

def _all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """
    Average the tensor across all distributed processes.
    Ensures memory prototypes are synchronized during distributed training.
    """
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor

class CategoryNeuronsMemory(nn.Module):
    """
    Explicit Multiscale Prototype Memory (DREAM Module).
    
    Key Features:
    - Read Mechanism: Uses an isolated snapshot (fp32) as Keys/Values.
      Computes retrieval via nn.MultiheadAttention in fp32 to ensure stability.
    - Write Mechanism: Updates internal prototypes (self.memory) via an 
      Exponential Moving Average (EMA) strategy without breaking the autograd graph.
    - Mixed Precision Safety: Internal computations strictly run in float32. 
      Inputs/Outputs automatically align with the external data type (e.g., fp16/bf16).
    """
    def __init__(
        self,
        memory_size: int = 2048,
        feature_dim: int = 1024,
        ema: float = 1e-5,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        batch_first: bool = True,
        ema_decay: bool = True,
    ):
        super().__init__()
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.ema_base = float(ema)
        self.ema_decay = ema_decay

        # --- Internal FP32 Island ---
        # The memory bank is registered as a non-trainable parameter and updated via EMA
        self.register_parameter(
            "memory",
            nn.Parameter(torch.randn(memory_size, feature_dim, dtype=torch.float32), requires_grad=False)
        )

        self.mha = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=batch_first,
        )
        # Lock MHA parameters to float32 to maintain the fp32 computational island
        self.mha.to(torch.float32)

        # EMA decay scheduler
        if ema_decay:
            self.global_step = 0
            # Decay strategy: drops to 1/10th every 20,000 steps
            self.ema_k = math.log(10.0) / 20000.0 

        # Snapshot for read operations (refreshed before each forward pass)
        self._snapshot: Optional[torch.Tensor] = None

    def get_cur_ema(self) -> float:
        """Calculate the current EMA momentum based on the global step."""
        if not self.ema_decay:
            return self.ema_base
        return self.ema_base * math.exp(-self.ema_k * self.global_step)

    def start_step(self):
        """
        Refresh the memory snapshot.
        Highly recommended to be called externally at the beginning of each forward step.
        The snapshot operates on the same device and dtype (fp32) but is detached from autograd.
        """
        self._snapshot = self.memory.detach()

    def read(self, query: torch.Tensor, return_loss: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Memory Retrieval Mechanism.
        
        Args:
            query: Visual feature queries of shape (B, N, D).
            return_loss: Whether to compute and return the sparsity regularization loss.
            
        Returns:
            retrieved: The retrieved memory features (compositional factors) of shape (B, N, D).
            loss: (Optional) The sparsity regularization loss.
        """
        assert query.dim() == 3 and query.size(-1) == self.feature_dim, \
            f"Expected query shape (B, N, {self.feature_dim}), got {query.shape}"

        # Fallback: automatically generate a snapshot if not explicitly refreshed
        if self._snapshot is None:
            self.start_step()

        device = query.device
        out_dtype = query.dtype  # Preserve external dtype
        
        # Cast inputs to fp32 island
        q = query.to(torch.float32)
        mem = self._snapshot.to(torch.float32)
        mem_batched = mem.unsqueeze(0).expand(q.size(0), -1, -1).to(device)

        # Disable autocast to enforce pure fp32 computation internally
        with torch.autocast(device_type=device.type, enabled=False):
            attn_output, attn_weights = self.mha(
                q, mem_batched, mem_batched,
                need_weights=True,
                average_attn_weights=True
            )

        attn_output = _nan_to_num_(attn_output)

        # Safety check: Catch NaN anomalies inside the fp32 island
        if torch.isnan(attn_output).any():
            raise ValueError("NaN detected in attn_output inside the FP32 computational island.")

        if return_loss:
            loss = sparse_entropy_loss(attn_weights.reshape(-1, self.memory_size))
            return attn_output.to(out_dtype), loss.to(out_dtype)

        return attn_output.to(out_dtype)


    @torch.no_grad()
    def write(self, query: torch.Tensor, value: torch.Tensor):
        """
        Prototype Memory Consolidation (Write Mechanism).
        
        Updates the memory prototypes via an attention-weighted Exponential Moving 
        Average (EMA) strategy. It executes entirely in fp32 under a no_grad 
        context to prevent autograd graph disruptions and mixed-precision instability.
        
        Args:
            query: Visual feature queries of shape (B, N, D).
            value: Compositional values to be written, shape (B, N, D).
        """
        assert query.shape == value.shape, "Query and Value shapes must match."
        assert query.dim() == 3 and query.size(-1) == self.feature_dim

        # Fallback: automatically generate a snapshot if not explicitly refreshed
        if self._snapshot is None:
            self.start_step()

        device = query.device
        
        # 1. Cast inputs to the FP32 computational island
        q = query.to(torch.float32)
        v = value.to(torch.float32)
        mem = self._snapshot.to(torch.float32)
        
        # Batch expansion for memory prototypes
        mem_batched = mem.unsqueeze(0).expand(q.size(0), -1, -1).to(device)

        # 2. Compute attention weights
        # Ensure MHA is in eval mode during write to freeze dropout, maintaining consistency with read
        old_training_state = self.mha.training
        self.mha.eval()
        try:
            with torch.autocast(device_type=device.type, enabled=False):
                _, attn_weights = self.mha(
                    q, mem_batched, mem_batched,
                    need_weights=True,
                    average_attn_weights=True
                )
        finally:
            self.mha.train(old_training_state)

        # Sanitize outputs within the FP32 island
        attn_weights = _nan_to_num_(attn_weights).to(torch.float32).contiguous()
        v = _nan_to_num_(v).to(torch.float32).contiguous()

        # 3. Calculate memory delta via attention-weighted aggregation
        # Step 3a: Calculate the weighted sum
        delta = torch.einsum('bnm,bnd->md', attn_weights, v)
        
        # Step 3b: Calculate the total weight assigned to each memory prototype
        attn_sum = attn_weights.sum(dim=(0, 1))  # Shape: (M,)
        
        # Step 3c: Normalize to get the weighted average (add epsilon to prevent division by zero)
        # This prevents the memory values from exploding when sequence lengths are large
        delta = delta / (attn_sum.unsqueeze(-1) + 1e-9)
        delta = _nan_to_num_(delta)

        # 4. Synchronize delta across distributed GPUs
        delta = _all_reduce_mean_(delta)

        # 5. EMA scheduling
        if self.ema_decay:
            self.global_step += 1
            cur_ema = float(self.get_cur_ema())
        else:
            cur_ema = float(self.ema_base)

        # 6. Apply EMA Update safely
        # Modifying .data directly bypasses PyTorch's in-place version tracking.
        # This prevents "RuntimeError: modified by an inplace operation" during backward().
        self.memory.data = (1.0 - cur_ema) * self.memory.data + cur_ema * delta