"""Request-local H3 noise draws, in the released model's exact order and row layout."""

from __future__ import annotations

from typing import Any


def condition_noise(generator: Any, latents: Any) -> Any:
    """Draw one keyframe's host noise and place it beside the encoded anchor."""
    import torch

    return torch.randn(
        latents.shape,
        generator=generator,
        dtype=torch.float32,
    ).to(device=latents.device, dtype=latents.dtype)


def initial_latents(generator: Any, config: Any, grid: Any, *, device: Any) -> dict[str, Any]:
    """Draw target video then official channel-major audio rows from the same stream."""
    import torch

    from .dit import unpack_audio

    video = torch.randn(
        1,
        config.latents_dim,
        grid.latent_t,
        grid.latent_h,
        grid.latent_w,
        generator=generator,
    ).to(device)
    audio_rows = torch.randn(
        grid.audio_rows,
        config.audio_latents_dim,
        generator=generator,
    )
    return {"video": video, "audio": unpack_audio(audio_rows).to(device)}
