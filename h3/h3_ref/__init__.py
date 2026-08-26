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
everything it adds is the part Cozy owns and upstream does not: which components an
artifact has, and building each one EMPTY so the runtime's fill plane can put our bytes in
it. Weights never arrive over the network here — no hub call, no offload manager, no
device choreography, no `.to()`. Those are `cozy_runtime`'s, exactly as in `h3_arch/`.

WHAT IS PROVEN, AND AT WHICH GRADE. `scripts/h3-diffusers-keys.py` builds all four
weight-bearing components on `meta` and diffs their census against the banked headers of
the official release's diffusers-format tree. The result is EXACT IDENTITY — 638
transformer, 703 video-VAE, 1087 audio-VAE and 1058 text-encoder destinations, zero
renames, zero shape or dtype disagreements. That is header-verified evidence, on the
control plane, for $0. It is NOT output verification: no number here has been produced on
a card.

`rope.inv_freq` is the one tensor #508c named as the native/diffusers delta, and the census
shows why it is not a delta at all: upstream registers it NON-PERSISTENTLY, so it is absent
from the state dict and absent from the banked safetensors header alike. It is derived at
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
    "COMPONENTS",
    "build_audio_vae",
    "build_component",
    "build_scheduler",
    "build_text_encoder",
    "build_transformer",
    "build_video_vae",
    "component_config",
    "config_mapping",
]

#: The artifact's components, and the ONLY names this package answers to. They are the
#: endpoint's component vocabulary (`h3.py`'s `H3Pipeline.components`), deliberately
#: unchanged by the rebase: the runtime-facing contract is the thing that must not move
#: while the internals become upstream's. Upstream's own tree spells the video VAE `vae`; the
#: translation is one row of `_SUBFOLDER` and never leaks into a binding or a descriptor.
COMPONENTS = ("transformer", "transformer_ref", "text_encoder", "video_vae", "audio_vae")

#: component -> the subfolder an official diffusers-format snapshot carries it under. This
#: is the PACKAGING's vocabulary, not ours, and it is written down once so that a config
#: mapping keyed either way resolves the same.
_SUBFOLDER: dict[str, str] = {
    "transformer": "transformer",
    "transformer_ref": "transformer_ref",
    "text_encoder": "text_encoder",
    "video_vae": "vae",
    "audio_vae": "audio_vae",
}


def component_config(component: str, mapping: dict[str, Any]) -> dict[str, Any]:
    """One component's construction kwargs out of the artifact's immutable config.

    A diffusers-format artifact's config IS the constructor's kwargs — that is what
    `config.json` in each subfolder holds — so there is no translation table here and no
    second authority for a hidden size. Three things are dropped and nothing else:

      * `_`-prefixed keys, which are packaging bookkeeping (`_class_name`,
        `_diffusers_version`) and not architecture;
      * keys the class does not accept, because a newer release naming a field this pin
        does not read is a compatibility fact rather than a caller error;
      * nothing else. An unrecognized VALUE is never silently corrected.

    The mapping may be keyed by COMPONENT (`video_vae`) or by SUBFOLDER (`vae`); both are
    the same component and both resolve.
    """
    if component not in COMPONENTS:
        raise KeyError(f"{component!r} is not an H3 component: {', '.join(COMPONENTS)}")
    section = mapping.get(component)
    if not isinstance(section, dict):
        section = mapping.get(_SUBFOLDER[component])
    if not isinstance(section, dict):
        return {}
    return {k: v for k, v in section.items() if not k.startswith("_")}


def config_mapping(whole: Any) -> dict[str, dict[str, Any]]:
    """`H3Config` -> the artifact-config mapping `component_config` reads.

    THE SEAM #540 DEFERRED, and it is a translation rather than a lookup: the two layers
    parameterize the same released facts differently, and every line below is one of those
    differences written down once. Upstream's own defaults hold wherever `H3Config` states
    nothing — a field this port never had is not a field to invent a value for.

    It exists so the endpoint can hand EITHER construction layer the same typed object.
    `H3Config` is the artifact's config as this repo types it; a diffusers-format artifact
    carries per-subfolder `config.json` documents instead, and `component_config` already
    reads those verbatim. So this is the NATIVE-config route into the upstream classes, and
    a real diffusers artifact bypasses it entirely.
    """
    dit, te, vv, av = whole.dit, whole.text_encoder, whole.video_vae, whole.audio_vae
    return {
        "transformer": {
            "num_attention_heads": dit.num_attention_heads,
            "attention_head_dim": dit.attention_head_dim,
            "hidden_size": dit.hidden_size,
            "num_layers": dit.num_layers,
            "num_refiner_layers": dit.token_refiner_num_layers,
            "ffn_dim": dit.ffn_hidden_size,
            "in_channels": dit.latents_dim,
            "audio_in_channels": dit.audio_latents_dim,
            "patch_size": tuple(dit.patch_size),
            "text_dim": dit.text_dim,
            "freq_dim": dit.timestep_input_dim,
            "time_embed_hidden_dim": dit.time_embed_hidden_size,
            "time_embed_dim": dit.time_embed_dim,
            "rope_freq_dim": dit.rope_inv_freq_len,
            "norm_eps": dit.norm_eps,
            "qk_norm_eps": dit.qk_norm_eps,
            "final_norm_eps": dit.final_norm_eps,
            # the two terms upstream's class does not know about; `build_transformer` pops
            # them and they select a topology rather than a numeric default
            "structure": dit.structure.value,
            "adaln_curve_grid": dit.adaln_curve_grid,
        },
        "text_encoder": {
            "model_type": "qwen3_vl",
            "num_hidden_layers": te.num_hidden_layers,
            "hidden_size": te.hidden_size,
            "intermediate_size": te.intermediate_size,
            "num_attention_heads": te.num_attention_heads,
            "num_key_value_heads": te.num_key_value_heads,
            "head_dim": te.head_dim,
            "vocab_size": te.vocab_size,
            "rms_norm_eps": te.rms_norm_eps,
            "rope_theta": te.rope_theta,
        },
        "video_vae": {
            "in_channels": vv.in_channels,
            "out_channels": vv.out_ch,
            "latent_channels": vv.embed_dim,
            # upstream states the per-stage WIDTHS where this port states a base and a
            # multiplier table; the widths are the product and there is no third authority
            "block_out_channels": tuple(vv.ch * m for m in vv.ch_mult),
            "layers_per_block": vv.num_res_blocks,
            "spatial_downsample_factors": tuple(vv.space_down),
            "temporal_downsample_factors": tuple(vv.time_down),
            "spatial_padding_mode": vv.padding_mode,
            "decoder_num_layers": vv.decoder_num_layers,
            "decoder_num_attention_heads": vv.decoder_heads,
            "decoder_attention_head_dim": vv.decoder_dim_head,
            # upstream carries the feed-forward as a MULTIPLE of the decoder width; this
            # port carries the width itself, and 8192 / 2048 is where the 4 comes from
            "decoder_ffn_mult": vv.decoder_ffn_dim // vv.decoder_dim,
            "decoder_rope_theta": vv.decoder_rope_theta,
            "decoder_rope_dim_ratio": vv.decoder_rope_dim_ratio,
            "decoder_norm_eps": vv.decoder_norm_eps,
            # #522b's temporal geometry, as the released config states it
            "clip_length": vv.vae_clip_length,
            "token_drop": vv.vae_token_drop,
        },
        "audio_vae": {
            "encoder_dim": av.encoder_dim,
            "encoder_rates": tuple(av.encoder_rates),
            "latent_dim": av.latent_dim,
            "latent_channels": av.latent_channels,
            "decoder_dim": av.decoder_dim,
            "decoder_rates": tuple(av.decoder_rates),
            "resblock_kernel_sizes": tuple(av.resblock_kernel_sizes),
            "resblock_dilation_sizes": tuple(tuple(d) for d in av.resblock_dilation_sizes),
            "sampling_rate": av.sample_rate,
        },
    }


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

    Both transformer components build this same class from the same config: the fl2va and
    ref2va partitions are structurally indistinguishable (job-001) and are told apart by the
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
    """The Qwen3-VL text encoder, upstream's own `Qwen3VLForConditionalGeneration`.

    This is the component #532 wants the GPL-adapted local `text_encoder.py` replaced by,
    and it is also the one place the rebase is not free: the official release ships the
    UNTRUNCATED text encoder (64 language layers, a vision tower and an LM head — 1058
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

    It carries no weights, so it is not a component and never appears in `COMPONENTS`: nothing
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


def build_component(component: str, config: dict[str, Any] | None = None) -> nn.Module:
    """One component root by its component name — the seam `scripts/h3-diffusers-keys.py`
    checks, and the same signature `h3_arch.build_component` answers to, so the pipeline
    switches implementations by changing an import and nothing else.

    `config` is the WHOLE artifact config mapping; this selects the component's section.
    A diffusers-format artifact's own per-subfolder `config.json` documents ARE that mapping;
    `config_mapping` produces the same shape from this repo's typed `H3Config` when the
    artifact carries no diffusers config of its own.
    """
    if component not in _BUILDERS:
        raise KeyError(f"{component!r} is not an H3 component: {', '.join(COMPONENTS)}")
    builder: Any = _BUILDERS[component]
    return builder(component_config(component, config or {}))
