# Apache-2.0; extracted from numz/ComfyUI-SeedVR2_VideoUpscaler. See NOTICE.
import torch

def blend_overlapping_frames(prev_tail: torch.Tensor, cur_head: torch.Tensor, overlap: int) -> torch.Tensor:
    """
    Blend two overlapping frame sequences in-place.
    
    Args:
        prev_tail: Last `overlap` frames from previous batch [overlap, H, W, C]
        cur_head: First `overlap` frames from current batch [overlap, H, W, C]
        overlap: Number of overlapping frames
        
    Returns:
        torch.Tensor: Blended frames [overlap, H, W, C]
    """
    device = prev_tail.device
    dtype = prev_tail.dtype
    
    # Smooth crossfade with Hann window for overlap >= 3, linear for smaller overlaps
    if overlap >= 3:
        t = torch.linspace(0.0, 1.0, steps=overlap, device=device, dtype=dtype)
        blend_start = 1.0 / 3.0
        blend_end = 2.0 / 3.0
        u = ((t - blend_start) / (blend_end - blend_start)).clamp(0.0, 1.0)
        w_prev_1d = 0.5 + 0.5 * torch.cos(torch.pi * u)  # Hann window
    else:
        w_prev_1d = torch.linspace(1.0, 0.0, steps=overlap, device=device, dtype=dtype)
    
    w_prev = w_prev_1d.view(overlap, 1, 1, 1)
    w_cur = 1.0 - w_prev
    
    return prev_tail * w_prev + cur_head * w_cur
