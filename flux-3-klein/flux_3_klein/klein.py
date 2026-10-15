"""FLUX.3 Image Klein, base and distilled, component-staged over Diffusers' modules.

The modules are Diffusers' (`Flux3Transformer2DModel`, `AutoencoderKLFlux2`, the Qwen3-VL prompt
encoding of its Flux3 modular pipeline); reference preparation and sampling follow BFL's reference
code (`black-forest-labs/flux-3-image-klein`, `flux3/pipeline.py`, `flux3/sampling.py`). Runtime
constructs and fills the modules and owns their residency.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from cozy_runtime.author import (
    Config,
    Context,
    Loader,
    Model,
    ModelArtifact,
    ModelAssets,
    Telemetry,
    UnsupportedInput,
    uses_components,
)
from cozy_runtime.derive.quantization import QuantizationSource
from diffusers import AutoencoderKLFlux2, Flux3Transformer2DModel
from diffusers.modular_pipelines.flux3.encoders import Flux3TextEncoderStep
from PIL import Image, ImageOps
from tensorfs.derived import Config as DerivedConfig
from tensorfs.derived import Derivation, Target
from tokenizers import Tokenizer
from transformers import (
    AutoModelForImageTextToText,
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLConfig,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)
from transformers import initialization as transformer_init

#: BFL's constant timestep shift, the image RoPE scale and Qwen3-VL's conditioning layers.
ALPHA = 6.93
IMAGE_ROPE_SCALE = 0.5
TEXT_LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)
PAD_MULTIPLE = 80
MAX_TOKENS = 8192
#: Pixels per latent token: the VAE's 8x, then 2x2 packing.
PATCH = 16
MATCH_MAX_PIXELS = 4 * 1024 * 1024
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024
MIN_SIDE, MAX_SIDE = 64, 4096
COMPONENTS = ("transformer", "text_encoder", "vae")


def _section(mapping: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = mapping[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"Klein {name} configuration must be a mapping")
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _conditioning_only(encoder: Any) -> None:
    """Klein reads hidden states 4..32 of Qwen3-VL's text stack: the vision tower, the last four
    layers, the final norm and the head are never run, so they are not built (their stored bytes
    are skipped). With the norm an identity, hidden state 32 stays the pre-norm output."""
    language = encoder.model.language_model
    language.layers = torch.nn.ModuleList(list(language.layers)[: max(TEXT_LAYERS)])
    language.norm = torch.nn.Identity()
    encoder.lm_head = torch.nn.Identity()
    del encoder.model.visual


def _conditioning_key(key: str) -> bool:
    """A stored text-encoder tensor `_conditioning_only` builds."""
    if key == "model.language_model.embed_tokens.weight":
        return True
    prefix = "model.language_model.layers."
    return key.startswith(prefix) and int(key[len(prefix):].split(".", 1)[0]) < max(TEXT_LAYERS)


def repack(source: QuantizationSource, ctx: Context, tel: Telemetry) -> ModelArtifact:
    """The served checkpoint from an as-is upload, by reference (no bytes copied): the three
    Diffusers components without BFL's `variants/` folder or the text-encoder tensors Klein never
    runs. Configs, tokenizer and processor files carry over."""
    capability = ctx.tensorfs_source(source)
    inspection = capability.inspect()
    missing = [name for name in COMPONENTS if name not in inspection.components]
    if missing:
        raise UnsupportedInput(f"the Klein upload lacks components {missing}")
    unused = tuple(sorted(k for k in inspection.components["text_encoder"] if not _conditioning_key(k)))
    targets = {
        name: Target(source="source", source_component=name, drop=unused if name == "text_encoder" else ())
        for name in COMPONENTS
    }
    configs = {
        name: DerivedConfig("copy", source="source", source_config=name)
        for name in inspection.configs
        if not name.startswith("variants")
    }
    order = tuple(
        (name, key)
        for name in COMPONENTS
        for key in inspection.components[name]
        if name != "text_encoder" or _conditioning_key(key)
    )
    with ctx.output("model").open(Derivation({"source": capability}, targets, configs, order)) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        tel.log("klein repacked", dropped_text_encoder=len(unused), configs=", ".join(sorted(configs)))
        return ctx.adopt_model(transaction.commit())


class KleinGraph:
    def __init__(self, config: Config) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            transformer = Flux3Transformer2DModel.from_config(_section(mapping, "transformer"))
            text_encoder = AutoModelForImageTextToText.from_config(
                Qwen3VLConfig(**_section(mapping, "text_encoder")), dtype=torch.bfloat16
            )
            vae = AutoencoderKLFlux2.from_config(_section(mapping, "vae"))
        _conditioning_only(text_encoder)
        self.components: dict[str, Any] = {
            "transformer": transformer.to(dtype=torch.bfloat16),
            "text_encoder": text_encoder,
            "vae": vae.to(dtype=torch.float32),
        }


def build_processor(assets: ModelAssets) -> Qwen3VLProcessor:
    """Qwen3-VL's processor from the checkpoint's `text_encoder/` files, parsed in memory."""

    def read(name: str) -> bytes:
        return assets.read(f"text_encoder/{name}")

    tokenizer_config = json.loads(read("tokenizer_config.json"))
    for key in ("tokenizer_file", "name_or_path", "added_tokens_decoder", "tokenizer_class",
                "processor_class", "chat_template"):
        tokenizer_config.pop(key, None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(read("tokenizer.json").decode()), **tokenizer_config
    )
    image = json.loads(read("preprocessor_config.json"))
    video = json.loads(read("video_preprocessor_config.json"))
    for config in (image, video):
        for key in ("processor_class", "image_processor_type", "video_processor_type"):
            config.pop(key, None)
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**image),
        video_processor=Qwen3VLVideoProcessor(**video),
        chat_template=json.loads(read("chat_template.json"))["chat_template"],
    )


def _ceil16(value: int) -> int:
    return -(-value // PATCH) * PATCH


def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """(fitted size, canvas) of a lone reference: never upscaled, the canvas (the fit rounded up to
    16) at most 4 MiP with sides 64-4096."""
    scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
    while True:
        fit = (max(1, round(width * scale)), max(1, round(height * scale)))
        canvas = (min(MAX_SIDE, max(MIN_SIDE, _ceil16(fit[0]))), min(MAX_SIDE, max(MIN_SIDE, _ceil16(fit[1]))))
        if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
            return fit, canvas
        scale *= 0.995


def prepare_reference(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """A lone reference, reflect-padded to its canvas; with the fitted image's box when padded."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    fit, (w, h) = reference_canvas(image.width, image.height)
    if fit != image.size:
        image = image.resize(fit, Image.Resampling.LANCZOS)
    if (w, h) == image.size:
        return image, None
    left, top = (w - image.width) // 2, (h - image.height) // 2
    padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
    pixels = np.asarray(image)
    padded = Image.fromarray(
        np.pad(pixels, padding, mode="reflect") if min(image.size) > PATCH
        else np.pad(pixels, padding, mode="symmetric")
    )
    return padded, (left, top, left + image.width, top + image.height)


def prepare_multi_reference(image: Image.Image) -> Image.Image:
    """One of several references: under 1 MP, centre-cropped to the 16-pixel grid."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width * height > MULTI_REFERENCE_MAX_PIXELS:
        scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
        width, height = int(width * scale), int(height * scale)
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    w, h = width // PATCH * PATCH, height // PATCH * PATCH
    left, top = (width - w) // 2, (height - h) // 2
    return image.crop((left, top, left + w, top + h))


def image_ids(rows: int, columns: int, layer: int = 0) -> torch.Tensor:
    """(t, h, w, l) position ids of a token grid; references take l = -1, -2, ..."""
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(rows), torch.arange(columns), torch.tensor([layer])
    ).float()
    ids[:, 1:3] *= IMAGE_ROPE_SCALE
    return ids


def text_ids(length: int) -> torch.Tensor:
    return torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(length)).float()


def schedule(steps: int) -> list[float]:
    t = np.linspace(1.0, 0.0, steps + 1)
    return [float(v) for v in ALPHA * t / (1.0 + (ALPHA - 1.0) * t)]


def _bn(vae: Any, like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = vae.bn.running_mean.view(1, -1, 1, 1).to(like)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(like)
    return mean, std


class Klein(Model[KleinGraph]):
    """One Klein checkpoint: the distilled flow map (no guidance) or the base model (CFG)."""

    graph: KleinGraph
    processor: Qwen3VLProcessor

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(KleinGraph, factory=KleinGraph)
        self.processor = build_processor(loader.assets)

    @property
    def flow_map(self) -> bool:
        return bool(self.graph.components["transformer"].config.flow_map)

    @uses_components("text_encoder", memoize=True, memo_dependencies=(build_processor, _conditioning_only))
    def encode(self, prompt: str) -> torch.Tensor:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds = Flux3TextEncoderStep._get_qwen3_vl_prompt_embeds(
                encoder,
                self.processor,
                prompt,
                device=next(encoder.parameters()).device,
                max_sequence_length=MAX_TOKENS,
                pad_multiple=PAD_MULTIPLE,
                hidden_states_layers=TEXT_LAYERS,
            )
        return embeds.cpu()

    @uses_components("vae", memoize=True)
    def encode_references(self, references: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized packed latents of the prepared references as one token sequence, with ids."""
        vae = self.graph.components["vae"]
        device = next(vae.parameters()).device
        tokens, ids = [], []
        with torch.inference_mode():
            for index, reference in enumerate(references):
                pixels = torch.from_numpy(np.array(reference)).permute(2, 0, 1).float() / 127.5 - 1.0
                latents = vae.encode(pixels[None].to(device=device, dtype=vae.dtype)).latent_dist.mode()
                b, c, h, w = latents.shape
                latents = latents.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4)
                latents = latents.reshape(b, c * 4, h // 2, w // 2)
                mean, std = _bn(vae, latents)
                latents = (latents - mean) / std
                tokens.append(latents.flatten(2).transpose(1, 2).to(torch.bfloat16).cpu())
                ids.append(image_ids(h // 2, w // 2, -(1 + index)))
        return torch.cat(tokens, dim=1), torch.cat(ids)

    @uses_components("transformer")
    def denoise(
        self,
        embeds: torch.Tensor,
        negative: torch.Tensor | None,
        references: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        width: int,
        height: int,
        steps: int,
        guidance: float | None,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
    ) -> torch.Tensor:
        """BFL's Euler sampler: a flow map jumps to the next timestep; the base model takes CFG."""
        transformer = self.graph.components["transformer"]
        device = next(transformer.parameters()).device
        rows, columns = height // PATCH, width // PATCH
        noise = torch.randn((rows * columns, transformer.config.in_channels), generator=torch.Generator().manual_seed(seed))
        x = noise.to(torch.bfloat16).float().to(device)[None]  # bf16 values, an fp32 state across steps
        fixed: dict[str, Any] = {"img_ids": image_ids(rows, columns).to(device)}
        if references is not None:
            fixed["image_cond_hidden_states"] = references[0].to(device)
            fixed["image_cond_ids"] = references[1].to(device)
        texts = [embeds] if negative is None else [negative, embeds]
        texts = [(t.to(device=device, dtype=torch.bfloat16), text_ids(t.shape[1]).to(device)) for t in texts]
        timesteps = schedule(steps)
        with torch.inference_mode():
            for step in range(steps):
                cancel()
                t, nxt = timesteps[step], timesteps[step + 1]
                now = torch.full((1,), t, device=device, dtype=torch.bfloat16)
                extra = {}
                if self.flow_map:
                    extra["target_timestep"] = torch.full((1,), nxt, device=device, dtype=torch.bfloat16)
                velocities = [
                    transformer(
                        hidden_states=x.to(torch.bfloat16),
                        encoder_hidden_states=text,
                        txt_ids=ids,
                        timestep=now,
                        return_dict=False,
                        **fixed,
                        **extra,
                    )[0].float()
                    for text, ids in texts
                ]
                if guidance is None:
                    (velocity,) = velocities
                else:
                    unconditional, conditional = velocities
                    velocity = unconditional + guidance * (conditional - unconditional)
                x = x + (nxt - t) * velocity
                on_step(step)
        return x.cpu()

    @uses_components("vae")
    def decode(self, tokens: torch.Tensor, *, width: int, height: int) -> Image.Image:
        vae = self.graph.components["vae"]
        device = next(vae.parameters()).device
        rows, columns = height // PATCH, width // PATCH
        with torch.inference_mode():
            latents = tokens.to(device=device, dtype=vae.dtype).view(1, rows, columns, -1).permute(0, 3, 1, 2)
            mean, std = _bn(vae, latents)
            latents = latents * std + mean
            b, c, h, w = latents.shape
            latents = latents.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
            image = vae.decode(latents, return_dict=False)[0][0]
            if not torch.isfinite(image).all():
                raise ValueError("Klein's decoder produced non-finite pixels")
            pixels = ((image.float().clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)
