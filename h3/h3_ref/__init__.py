"""The OFFICIAL MiniMax H3 implementation, as the endpoint's reference architecture.

WHY THIS EXISTS AND `h3_arch/` DOES NOT NEED TO (decision #531). The 3,100-line port next
door was written on a premise a source census would have falsified: that no maintained
implementation of H3 existed to depend on. Diffusers merged the official one on Aug 5
(PR #14355) — transformer, both VAEs, the H3-specific scheduler, and the text conditioning
— twenty days before the port landed, and it gets right, in upstream source, every seam
#522 found the port had got wrong:

  * the velocity is DATA-WARD (`x0 = x_t + sigma*v`), stated in the scheduler's own class
    docstring as the reason it cannot be a `FlowMatchEulerDiscreteScheduler` config;
  * the sigma grid is `linspace(1, 0, steps)` shifted and deduplicated, so 30 grid points
    drive 29 model evaluations, and `timesteps` is `1 - sigmas[:-1]` — points and
    evaluations are separately named rather than conflated;
  * the VAE's 17k+5 temporal geometry is the released `clip_length`/`token_drop` pair;
  * the decode returns pixels the caller scales once.

So this package is not a port. It is a THIN CONSTRUCTION LAYER over upstream classes, and
everything it adds is the part Cozy owns and upstream does not: which component roles an
artifact has, and building each one EMPTY so the runtime's fill plane can put our bytes in
it. Weights never arrive over the network here — no hub call, no offload manager, no
device choreography, no `.to()`. Those are `cozy_runtime`'s, exactly as in `h3_arch/`.

WHAT IS PROVEN, AND AT WHICH GRADE. `scripts/h3-diffusers-keys.py` builds all four
weight-bearing roles on `meta` and diffs their census against the banked headers of the
official release's diffusers-format tree. The result is EXACT IDENTITY — 638 transformer,
703 video-VAE, 1087 audio-VAE and 1058 conditioner destinations, zero renames, zero shape
or dtype disagreements. That is header-verified evidence, on the control plane, for $0.
It is NOT output verification: no number here has been produced on a card.

`rope.inv_freq` is the one tensor #508c named as the native/diffusers delta, and the census
shows why it is not a delta at all: upstream registers it NON-PERSISTENTLY, so it is absent
from the state dict and absent from the artifact header alike. It is derived at
construction, never filled.

MODULE SCOPE STAYS LIGHT, for the same reason `h3_arch/` keeps it light: `describe` runs in
a container with no GPU, no CUDA image and no weights, so every heavy import is inside the
builder that needs it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "ROLES",
    "build_audio_vae",
    "build_component",
    "build_scheduler",
    "build_text_encoder",
    "build_transformer",
    "build_video_vae",
    "component_config",
]

#: The artifact's component roles, and the ONLY names this package answers to. They are the
#: endpoint's role vocabulary (`h3.py`'s `H3Pipeline.components`), deliberately unchanged by
#: the rebase: the runtime-facing contract is the thing that must not move while the
#: internals become upstream's. Upstream's own tree spells the video VAE `vae`; the
#: translation is one row of `_SUBFOLDER` and never leaks into a binding or a descriptor.
ROLES = ("transformer", "transformer_ref", "text_encoder", "video_vae", "audio_vae")

#: role -> the subfolder an official diffusers-format snapshot carries it under. This is the
#: PACKAGING's vocabulary, not ours, and it is written down once so that a config mapping
#: keyed either way resolves the same.
_SUBFOLDER: dict[str, str] = {
    "transformer": "transformer",
    "transformer_ref": "transformer_ref",
    "text_encoder": "text_encoder",
    "video_vae": "vae",
    "audio_vae": "audio_vae",
}


def component_config(role: str, mapping: dict[str, Any]) -> dict[str, Any]:
    """One role's construction kwargs out of the artifact's immutable config.

    A diffusers-format artifact's config IS the constructor's kwargs — that is what
    `config.json` in each subfolder holds — so there is no translation table here and no
    second authority for a hidden size. Three things are dropped and nothing else:

      * `_`-prefixed keys, which are packaging bookkeeping (`_class_name`,
        `_diffusers_version`) and not architecture;
      * keys the class does not accept, because a newer release naming a field this pin
        does not read is a compatibility fact rather than a caller error;
      * nothing else. An unrecognized VALUE is never silently corrected.

    The mapping may be keyed by ROLE (`video_vae`) or by SUBFOLDER (`vae`); both are the
    same component and both resolve.
    """
    if role not in ROLES:
        raise KeyError(f"{role!r} is not an H3 component role: {', '.join(ROLES)}")
    section = mapping.get(role)
    if not isinstance(section, dict):
        section = mapping.get(_SUBFOLDER[role])
    if not isinstance(section, dict):
        return {}
    return {k: v for k, v in section.items() if not k.startswith("_")}


def _accepted(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Kwargs the class actually declares. Upstream's `ConfigMixin` stores unknown keys
    instead of refusing them, which would turn a typo into a silently different graph."""
    import inspect

    names = set(inspect.signature(cls.__init__).parameters) - {"self"}
    return {k: v for k, v in kwargs.items() if k in names}


#: The one structural term an H3 artifact's config states that upstream's class does not
#: know about (§1.1.1). It is not a knob: getting it wrong is a different key set, which is
#: why it selects a topology here and never a numeric default.
CURVE_STRUCTURE = "h3-adaln-curve"


def build_transformer(config: dict[str, Any] | None = None) -> nn.Module:
    """The packed-token audio-video DiT, `MiniMaxH3Transformer3DModel`.

    Both transformer roles build this same class from the same config: the fl2va and ref2va
    partitions are structurally indistinguishable (job-001) and are told apart by the
    recipe's declared content digest, never by their shapes.

    TWO TOPOLOGIES, ONE CLASS. An artifact whose config names the curve structure gets the
    community-pruned modulation installed over the same class before it is censused — see
    `curve.py`, which is the whole of that difference. Everything else, including every key
    the fill plane will match, is upstream's.
    """
    from diffusers import MiniMaxH3Transformer3DModel

    kwargs = dict(config or {})
    structure = kwargs.pop("structure", None)
    grid = int(kwargs.pop("adaln_curve_grid", 0) or 0)
    transformer = MiniMaxH3Transformer3DModel(**_accepted(MiniMaxH3Transformer3DModel, kwargs))
    if structure == CURVE_STRUCTURE:
        from .curve import CURVE_GRID, install

        install(transformer, grid=grid or CURVE_GRID)
    return transformer


def build_text_encoder(config: dict[str, Any] | None = None) -> nn.Module:
    """The Qwen3-VL conditioner, upstream's own `Qwen3VLForConditionalGeneration`.

    This is the component #532 wants the GPL-adapted local `text_encoder.py` replaced by,
    and it is also the one place the rebase is not free: the official release ships the
    UNTRUNCATED conditioner (64 language layers, a vision tower and an LM head — 1058
    destinations, 66.7 GB), where the community repackaging ships it cut to 50 layers with
    no head. Same architecture, more resident bytes, and the residency ladder is entitled to
    know that before anyone rents a card for it.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText

    kwargs = dict(config or {})
    model_type = kwargs.pop("model_type", "qwen3_vl")
    conf = AutoConfig.for_model(model_type, **{k: v for k, v in kwargs.items()})
    return AutoModelForImageTextToText.from_config(conf)


def build_video_vae(config: dict[str, Any] | None = None) -> nn.Module:
    """`AutoencoderKLMiniMaxH3` — the 3D causal encoder and ViT decoder.

    Its `clip_length` (17) and `token_drop` (3) ARE #522b's temporal geometry, carried as
    released config rather than as a constant this repo has to keep correct.
    """
    from diffusers import AutoencoderKLMiniMaxH3

    return AutoencoderKLMiniMaxH3(**_accepted(AutoencoderKLMiniMaxH3, config or {}))


def build_audio_vae(config: dict[str, Any] | None = None) -> nn.Module:
    """`AutoencoderKLMiniMaxH3Audio` — the 32 kHz DAC encoder and BigVGAN decoder."""
    from diffusers import AutoencoderKLMiniMaxH3Audio

    return AutoencoderKLMiniMaxH3Audio(**_accepted(AutoencoderKLMiniMaxH3Audio, config or {}))


def build_scheduler(config: dict[str, Any] | None = None) -> Any:
    """One `MiniMaxH3Scheduler`. H3 runs TWO per request — one per modality — because the
    exponential sigma shift differs (12.0 video, 3.0 audio) and the modality is a property
    of the schedule, not of the class. A pipeline holds two instances; this builds one.

    It carries no weights, so it is not a component and never appears in `ROLES`: nothing
    censuses it and nothing fills it.
    """
    from diffusers import MiniMaxH3Scheduler

    return MiniMaxH3Scheduler(**_accepted(MiniMaxH3Scheduler, config or {}))


_BUILDERS: dict[str, Any] = {
    "transformer": build_transformer,
    "transformer_ref": build_transformer,
    "text_encoder": build_text_encoder,
    "video_vae": build_video_vae,
    "audio_vae": build_audio_vae,
}


def build_component(role: str, config: dict[str, Any] | None = None) -> nn.Module:
    """One component root by its artifact role — the seam `scripts/h3-diffusers-keys.py`
    checks, and the same signature `h3_arch.build_component` answers to, so the pipeline
    switches implementations by changing an import and nothing else.

    `config` is the WHOLE artifact config mapping; this selects the role's section from it.
    """
    if role not in _BUILDERS:
        raise KeyError(f"{role!r} is not an H3 component role: {', '.join(ROLES)}")
    builder: Any = _BUILDERS[role]
    return builder(component_config(role, config or {}))
