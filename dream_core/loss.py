import torch

def sparse_entropy_loss(attention_weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate the sparsity and anti-collapse regularization loss.
    
    This loss enforces two properties on the memory retrieval mechanism:
    1. Sparsity (Local Entropy): Encourages each local region (query) to attend 
       to only a sparse set of prototypes.
    2. Anti-collapse (Global Entropy): Encourages uniform utilization of all 
       prototypes across the entire batch to prevent memory collapse.
       
    Args:
        attention_weights: Tensor of shape (..., M), where M is the memory size.
        
    Returns:
        loss: A scalar tensor containing the calculated regularization loss.
    """
    orig_dtype = attention_weights.dtype
    
    # Cast to float32 to prevent underflow/overflow during log calculations
    p = attention_weights.to(torch.float32).clamp_min(1e-12)
    
    # 1. Local Entropy: Minimize entropy per query to promote sparsity
    entropy_per_loc = -(p * p.log()).sum(dim=-1).mean()
    
    # 2. Global Entropy: Maximize global entropy to promote selective activation
    p_mean = p.mean(dim=0)
    entropy_global = -(p_mean * p_mean.log()).sum()
    
    # Total loss: local entropy minus global entropy
    loss = entropy_per_loc - entropy_global
    
    return loss.to(orig_dtype)