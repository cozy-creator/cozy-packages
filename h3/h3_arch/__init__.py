"""The MiniMax H3 architecture, as an ORDINARY MODEL LIBRARY this endpoint brings.

cozy-runtime defines no model architecture (§1.1), and an endpoint that borrowed one from
the runtime's fixture closure would pin its topology to somebody else's test suite
(se-008's divergence 1). SDXL's answer was to depend on `diffusers`; H3's cannot be, and
the reason is measured rather than preferred: the artifact job-001 produced is the
COMMUNITY CURVE topology — 532 logical keys in `Comfy-Org/MiniMax-H3`'s own key dialect
(`753aab65…`) — and the only packaging `diffusers` can construct is the official 638-key
one (`0f112591…`). Two dialects, no rekey between them, because the curve topology has no
Diffusers counterpart at all: its AdaLN branch is a different SHAPE, not a different name.

So the architecture is here, ported from the loader proto-001 actually benchmarked the
selected candidate under (ComfyUI v0.33.0, `comfy/ldm/minimax/`), with everything that is
not architecture removed:

  * `comfy.ops` / `operations.Linear`  ->  plain `torch.nn`. The injectable-ops indirection
    exists to let a loader decide dtype and casting per weight, which is the fill plane's
    decision here and not an author's.
  * `comfy.model_management.cast_to`  ->  deleted. A cast at use-time is device management.
  * `comfy.model_prefetch`, `comfy.patcher_extension`  ->  deleted. Prefetch is staging and
    a patcher wrapper is an adapter runtime; both are runtime capabilities (§3.2).
  * `comfy.quant_ops`  ->  deleted. Runtime quantization is a DELETED surface (se-001).
  * `optimized_attention`  ->  `torch.nn.functional.scaled_dot_product_attention`, with the
    expert kernel chosen in `attention.py` as ordinary endpoint code (kernel choice is
    author territory; placement is not).

What survives is the numerics and the key names, and both are checked rather than trusted:
`scripts/h3-keys.py` builds every component on `meta` and diffs its census against the
pinned upstream header, which is the whole reason this file can claim a port at all.

MODULE SCOPE STAYS LIGHT. `import torch` happens inside the builders, never here: the
descriptor container has no CUDA and no weights, and `describe` must still run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import (
    DIT_FULL,
    AdaLnStructure,
    AudioVaeConfig,
    DitConfig,
    H3Config,
    Task,
    TextEncoderConfig,
    VideoVaeConfig,
)

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "DIT_FULL",
    "AdaLnStructure",
    "AudioVaeConfig",
    "DitConfig",
    "H3Config",
    "Task",
    "TextEncoderConfig",
    "VideoVaeConfig",
    "build_audio_vae",
    "build_component",
    "build_dit",
    "build_text_encoder",
    "build_video_vae",
]


def build_dit(config: DitConfig | None = None) -> nn.Module:
    from .dit import MiniMaxH3Dit

    return MiniMaxH3Dit(config or DitConfig())


def build_text_encoder(config: TextEncoderConfig | None = None) -> nn.Module:
    from .text_encoder import Qwen3VLConditioner

    return Qwen3VLConditioner(config or TextEncoderConfig())


def build_video_vae(config: VideoVaeConfig | None = None) -> nn.Module:
    from .video_vae import VideoVae

    return VideoVae(config or VideoVaeConfig())


def build_audio_vae(config: AudioVaeConfig | None = None) -> nn.Module:
    from .audio_vae import AudioVae

    return AudioVae(config or AudioVaeConfig())


#: Component ROLE -> builder. The two transformer roles build the same class from the same
#: config: they are structurally indistinguishable and are told apart by the recipe's
#: declared content digest, never by their shapes (job-001).
_BUILDERS: dict[str, Any] = {
    "transformer": build_dit,
    "transformer_ref": build_dit,
    "text_encoder": build_text_encoder,
    "video_vae": build_video_vae,
    "audio_vae": build_audio_vae,
}


def build_component(role: str, config: H3Config | None = None) -> nn.Module:
    """One component root by its artifact role — the seam `scripts/h3-keys.py` checks."""
    if role not in _BUILDERS:
        raise KeyError(f"{role!r} is not an H3 component role: {', '.join(_BUILDERS)}")
    whole = config or H3Config()
    per_role = {
        "transformer": whole.dit,
        "transformer_ref": whole.dit,
        "text_encoder": whole.text_encoder,
        "video_vae": whole.video_vae,
        "audio_vae": whole.audio_vae,
    }
    builder: Any = _BUILDERS[role]
    return builder(per_role[role])
