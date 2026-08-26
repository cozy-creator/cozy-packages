"""The H3 architecture's CHECKPOINT-VERSION FACTS, typed.

Every number here is a fact about a released checkpoint, not a knob and not a preference.
They arrive from the artifact's immutable `Config` when it carries them and default to the
values the pinned release states about itself — `Comfy-Org/MiniMax-H3@4cc1d817…`'s own
safetensors header metadata for the three components that publish one, and the DiT's tensor
shapes for the one that does not.

Nothing here names a repo, a release, a checkpoint or a revision: a hidden size is not a
selection. The `structure` term is the exception that proves the rule — it is a STRUCTURAL
term (§1.1.1's three AdaLN structures), and getting it wrong is a different key set, which
is why it is a typed enum with no default that means "whatever the bytes are".
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any


class AdaLnStructure(enum.Enum):
    """§1.1.1's three structures. `CURVE` is a community-pruned topology with its own
    numeric semantics and its own quality approval — never shorthand for BAKED, and never
    an automatic degradation from FULL."""

    FULL = "h3-full"
    CURVE = "h3-adaln-curve"
    BAKED = "h3-adaln-baked"


class GraphDialect(enum.Enum):
    """WHICH implementation constructs the component roots — a STRUCTURAL term (§1.1.1),
    never a preference, because the two dialects answer to DIFFERENT KEY SETS and an
    artifact fills exactly one of them.

    Measured, not asserted: `scripts/h3-keys.py` censuses the native port at 3,445
    destinations against the community carrier (transformer 532, text encoder 902, video
    VAE 562, audio VAE 917) and `scripts/h3-diffusers-keys.py` censuses the upstream classes
    at 3,486 against the official diffusers-format tree (638 / 1058 / 703 / 1087). Every
    COMPONENT disagrees, and the text encoder disagrees structurally rather than by name — the
    community carrier is Qwen3-VL cut to 50 layers with no head, the official tree is the
    untruncated 64-layer model. So this term selects a graph AND declares which artifact can
    fill it; picking the wrong one is `incomplete_fill` on a rented card.
    """

    NATIVE = "h3-native"
    DIFFUSERS = "h3-diffusers"


class Task(enum.Enum):
    """The task partition a transformer component was trained for. Structurally invisible
    (job-001: the two carriers have byte-identical headers), so it is carried as a stamp
    and checked by declared digest, never inferred from the weights."""

    FL2VA = "fl2va"
    REF2VA = "ref2va"


@dataclass(frozen=True, slots=True)
class DitConfig:
    """The packed-token audio-video DiT.

    The CURVE defaults: `time_embed_dim=8` is the interpolated curve's basis width and
    `adaln_curve_grid=1025` its sampled grid, which together are why a curve block's
    `adaln_proj.linear` is `[96768, 8]` and a full block's would be `[96768, 2688]`. That
    single number is the 26 GB the pruned topology does not carry.
    """

    structure: AdaLnStructure = AdaLnStructure.CURVE
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 8
    adaln_curve_grid: int = 1025
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    sigma_shift_video: float = 12.0
    sigma_shift_audio: float = 3.0

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw


#: The FULL-AdaLN sibling of the same release, kept as a named structure rather than a
#: comment: it is the 535-key/538-key lane the red arms substitute in (job-001's
#: `UNREGISTERED_FINGERPRINT` arm), and naming it is what makes that arm spellable.
DIT_FULL = DitConfig(structure=AdaLnStructure.FULL, time_embed_dim=2688, adaln_curve_grid=0)


@dataclass(frozen=True, slots=True)
class TextEncoderConfig:
    """Qwen3-VL-32B TRUNCATED to 50 layers, plus its 27-block vision tower.

    `num_hidden_layers: 50` and `output: unnormalized_hidden_after_layer_50` are the
    release's own header metadata (`minimax_h3_te`), and they are ALL of it. The truncation
    is why there is no final norm and no LM head in the key set: layer 50's raw output IS
    the conditioning.

    EVERY OTHER FIELD HERE IS UPSTREAM QWEN3-VL's, not this release's own statement — the
    header carries no vision config, no rope config and no eps values. Several of them
    (the deepstack indices, the rope sections, the vision rope theta) change numerics
    without changing a single key, so a green key check says nothing about them.
    """

    num_hidden_layers: int = 50
    hidden_size: int = 5120
    intermediate_size: int = 25600
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5000000.0
    #: vision tower
    vision_depth: int = 27
    vision_hidden_size: int = 1152
    vision_intermediate_size: int = 4304
    vision_num_heads: int = 16
    vision_patch_size: int = 16
    vision_temporal_patch_size: int = 2
    vision_spatial_merge_size: int = 2
    vision_in_channels: int = 3
    vision_num_position_embeddings: int = 2304
    vision_out_hidden_size: int = 5120
    #: Where the vision tower's deepstack features are TAKEN (upstream Qwen3-VL's
    #: `deepstack_visual_indexes`). They are ADDED into the language stream at its first
    #: three decoder layers, not at these. The key set is identical for any triple, so this
    #: is one of the numerics `scripts/h3-keys.py` structurally cannot check.
    vision_deepstack_layers: tuple[int, ...] = (8, 16, 24)
    vision_norm_eps: float = 1e-6


@dataclass(frozen=True, slots=True)
class VideoVaeConfig:
    """The 3D causal-encoder / ViT-decoder video VAE, `vae_ratio` 16 spatial x 4 temporal.

    Values are the release header's `minimax_h3_video_vae.source_config`; the two
    `latents_*` statistics are TENSORS in the key set, not config, and are not restated
    here — a normalization constant that ships as a weight is a weight.
    """

    ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8)
    num_res_blocks: int = 2
    in_channels: int = 3
    out_ch: int = 3
    z_channels: int = 24
    embed_dim: int = 24
    space_down: tuple[int, ...] = (2, 2, 2, 2, 1, 1)
    time_down: tuple[int, ...] = (1, 2, 2, 1, 1, 1)
    padding_mode: str = "reflect"
    causal_encoder: bool = True
    use_3d_conv: bool = True
    use_t_isolated_gn: bool = True
    vae_ratio: int = 16
    vae_ratio_t: int = 4
    vae_clip_length: int = 17
    vae_token_drop: int = 3
    #: The DECODE WINDOW, in pixels. Not a memory budget: the ViT decoder's RoPE coordinates
    #: are normalized by the extent of the window it is called on, so the window size is a
    #: term of the function. 256 px is 16 latent cells; see `video_vae` for what a wider one
    #: does to the output.
    vae_tile_size: int = 256
    vae_tile_overlap_min: int = 64
    #: the ViT decoder (`vit_decoder_kwargs`)
    decoder_dim: int = 2048
    decoder_num_layers: int = 36
    decoder_heads: int = 32
    decoder_dim_head: int = 64
    decoder_ffn_dim: int = 8192
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    #: 1e-5, matching the reference loader. The release's `vit_decoder_kwargs` names
    #: `norm_type`, `qk_norm_type` and both affine flags and NO eps, so this is the
    #: benchmarked loader's constant and not the checkpoint's own statement.
    decoder_norm_eps: float = 1e-5


@dataclass(frozen=True, slots=True)
class AudioVaeConfig:
    """The 32 kHz stereo audio VAE — a DAC-style encoder and a BigVGAN decoder, with one
    attention `pre_block` between the encoder trunk and the latent projections."""

    sample_rate: int = 32000
    output_channel: int = 2
    latent_channels: int = 32
    latent_dim: int = 2048
    encoder_dim: int = 64
    encoder_rates: tuple[int, ...] = (2, 4, 4, 5, 5)
    decoder_dim: int = 1024
    decoder_rates: tuple[int, ...] = (5, 5, 2, 2, 2, 2, 2)
    attn_proj: bool = True
    #: BigVGAN's per-upsample residual kernel/dilation family
    resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: tuple[tuple[int, ...], ...] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    snake_logscale: bool = True


@dataclass(frozen=True, slots=True)
class H3Config:
    """One artifact's whole construction configuration — five components, one object."""

    dit: DitConfig = field(default_factory=DitConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    video_vae: VideoVaeConfig = field(default_factory=VideoVaeConfig)
    audio_vae: AudioVaeConfig = field(default_factory=AudioVaeConfig)
    graph: GraphDialect = GraphDialect.NATIVE
    """Which construction layer builds the roots. NATIVE is the default because it is what
    the ONE ingested H3 artifact carries: job-001's dual snapshot is the community curve
    carrier, and no diffusers-format artifact is bound anywhere yet (#540e)."""

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> H3Config:
        """Overlay whatever the artifact's immutable config states over the release facts.

        An artifact ingested `single_file.identity/1` carries the ORIGINAL header, whose
        three metadata blobs are the upstream configs; an artifact that carries none leaves
        every field at the release's own value. Unknown fields are ignored rather than
        refused: this is the artifact's config, not a request, and a newer release naming a
        field this code does not read is a compatibility fact, not a caller error.
        """
        out = cls()
        stated = mapping.get("graph")
        if stated is not None:
            try:
                out = replace(out, graph=GraphDialect(stated))
            except ValueError:
                raise ValueError(
                    f"{stated!r} is not an H3 graph dialect: "
                    + ", ".join(d.value for d in GraphDialect)
                    + " — this term selects a KEY SET, so an unknown value cannot default"
                ) from None
        for name, section in (
            ("dit", mapping.get("dit")),
            ("text_encoder", mapping.get("text_encoder")),
            ("video_vae", mapping.get("video_vae")),
            ("audio_vae", mapping.get("audio_vae")),
        ):
            if not isinstance(section, dict):
                continue
            current = getattr(out, name)
            known = {
                k: v for k, v in section.items() if k in type(current).__dataclass_fields__
            }
            out = replace(out, **{name: replace(current, **known)})
        return out
