"""
DREAM: Document Recognition with Explicit Adaptive Memory
CVPR 2026

This module provides the core plug-and-play explicit multiscale prototype memory 
components for multimodal document recognition architectures.
"""

from .prototype_memory_module import CategoryNeuronsMemory
from .loss import sparse_entropy_loss

__all__ = [
    "CategoryNeuronsMemory",
    "sparse_entropy_loss"
]
