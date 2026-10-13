"""Use the shared PyTorch attention adapter for both SeedVR2 architectures."""
from ..dit_v2.attention import FlashAttentionVarlen, TorchAttention

__all__ = ["FlashAttentionVarlen", "TorchAttention"]
