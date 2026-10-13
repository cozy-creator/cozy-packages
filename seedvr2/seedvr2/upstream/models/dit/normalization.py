"""Use the shared upstream norm selection with standard PyTorch fallbacks."""
from ..dit_v2.normalization import get_norm_layer, norm_layer_type

__all__ = ["get_norm_layer", "norm_layer_type"]
