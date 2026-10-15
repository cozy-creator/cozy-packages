"""Component-staged Qwen Image 2.1 over Diffusers' own pipeline.

The architecture and sampling are Diffusers' (`QwenImage21Pipeline`, pinned by commit until
it reaches a release). Runtime constructs and fills the modules and owns their residency;
no Diffusers weight loader, device placement, or offload hook takes part in serving.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import diffusers
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
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

from cozy_runtime import fusion
from cozy_runtime.author import (
    Config,
    ConformanceError,
    Loader,
    Model,
    ModelAssets,
    uses_components,
)
from cozy_runtime.author.kernels import Plan

try:
    from diffusers import (
        AutoencoderKLQwenImage21,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions
except ImportError:
    # A caller that only imports this package for its call proxies (H3) may hold an older
    # Diffusers. The module still imports there; building the model refuses.
    QwenImage21Pipeline = None

_Pipeline: Any = QwenImage21Pipeline or object


def _modulation(block: Any, arguments: Mapping[str, Any]) -> fusion.Modulation:
    """Diffusers' `_select_modulation_rows` as a row per token: a target-image token reads its
    sample's row, the text and condition-image prefix the trailing t=0 row."""
    table, mask = arguments["modulation"], arguments["target_token_mask"]
    batch, tokens = arguments["hidden_states"].shape[:2]
    samples = torch.arange(batch, device=table.device)[:, None]
    if mask is None:
        index = samples.expand(batch, tokens)
    else:
        index = torch.where(mask[None, :], samples, table.shape[0] - 1)
    return fusion.Modulation(index.reshape(-1), *table.chunk(4, dim=-1))


#: The 2.1 block as Runtime fuses it (scale, gate | scale, gate; tanh gates; no shift).
BLOCK = (
    fusion.DiTBlock(
        QwenImage21TransformerBlock,
        width=4096,
        head_dim=128,
        modulation=_modulation,
        norms=("img_norm1", "img_norm2"),
        norm="layer",
        attention="attn",
        arguments=(
            "attention_mask",
            "rotary_emb",
            "layer_cache",
            "kv_cache_mode",
            "cache_write_slice",
            "segments",
            "key_valid",
        ),
        mlp=fusion.GatedMLP("img_mlp", gate="gate_layer", up="proj", down="out"),
        rope=fusion.Rope("rotary_emb", interleaved=True),
        gate="tanh",
    )
    if QwenImage21Pipeline is not None
    else None
)


class _Progress[T]:
    """Pipeline iteration without tqdm: a disabled bar still starts a monitor thread, and
    Runtime reports progress through Telemetry."""

    def __init__(self, iterable: Iterable[T] | None = None, total: int | None = None) -> None:
        self.iterable = iterable

    def __iter__(self) -> Iterator[T]:
        if self.iterable is None:
            raise TypeError("progress has no iterable")
        return iter(self.iterable)

    def update(self, n: int = 1) -> None:
        pass

    def __enter__(self) -> _Progress[T]:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _section(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Qwen Image {name} configuration must be a mapping")
    return dict(value)


class QwenImage21Graph:
    """One native construction census, with source checkpoint precision preserved."""

    def __init__(self, config: Config) -> None:
        if QwenImage21Pipeline is None:
            raise ConformanceError(
                "Qwen Image 2.1 needs the Diffusers this package pins (QwenImage21Pipeline); "
                f"this environment has diffusers {diffusers.__version__}",
                code="qwen_image21_diffusers",
            )
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            self.components: dict[str, Any] = {
                "transformer": QwenImage21Transformer2DModel.from_config(
                    _section(mapping["transformer"], "transformer")
                ).to(dtype=torch.bfloat16),
                # Construction dtype leaves derived RoPE buffers in Transformers' F32.
                # Casting the completed module would round those frequencies to BF16.
                "text_encoder": AutoModelForImageTextToText.from_config(
                    Qwen3VLConfig(**_section(mapping["text_encoder"], "text_encoder")),
                    dtype=torch.bfloat16,
                ),
                "vae": AutoencoderKLQwenImage21.from_config(_section(mapping["vae"], "vae")).to(
                    dtype=torch.float32
                ),
            }
        # The pipeline reads hidden states only, so the 1.24 GB language-model head is not
        # built; Runtime skips its stored weight.
        self.components["text_encoder"].lm_head = torch.nn.Identity()
        self.scheduler_config = _section(mapping["scheduler"], "scheduler")


def build_qwen_image21(config: Config) -> QwenImage21Graph:
    return QwenImage21Graph(config)


def build_processor(assets: ModelAssets) -> Any:
    """Qwen3-VL's processor from the checkpoint's upstream `processor/` files, parsed in memory."""

    def read(name: str) -> bytes:
        return assets.read(f"processor/{name}")

    tokenizer_config = json.loads(read("tokenizer_config.json"))
    for key in (
        "tokenizer_file",
        "name_or_path",
        "added_tokens_decoder",
        "tokenizer_class",
        "processor_class",
    ):
        tokenizer_config.pop(key, None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(read("tokenizer.json").decode()), **tokenizer_config
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessor(**json.loads(read("preprocessor_config.json"))),
        video_processor=Qwen3VLVideoProcessor(**json.loads(read("video_preprocessor_config.json"))),
        chat_template=read("chat_template.jinja").decode(),
    )


class _StagedPipeline(_Pipeline):
    reference_latents: torch.Tensor | None = None
    reference_pad_mask: torch.Tensor | None = None

    def progress_bar[T](
        self, iterable: Iterable[T] | None = None, total: int | None = None
    ) -> _Progress[T]:
        return _Progress(iterable, total)

    @property
    def _execution_device(self) -> torch.device:
        # Diffusers chooses the first registered module (VAE), which can correctly
        # remain parked on CPU while Runtime admits the denoiser. Follow the active
        # denoising component instead. Encoding supplies its own leased device.
        return next(self.transformer.parameters()).device

    def encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        # The denoiser must reuse the vision token positions from the separately
        # leased text encoder. Diffusers otherwise rejects precomputed image embeds.
        if self.reference_pad_mask is not None:
            kwargs["image_pad_mask"] = self.reference_pad_mask
        return super().encode_prompt(*args, **kwargs)

    def prepare_latents(self, images: Any, *args: Any, **kwargs: Any) -> Any:
        # References were encoded while the VAE had its own residency lease.
        # Never invoke that parked component from the transformer-only lease.
        latents, image_latents = super().prepare_latents(
            None if self.reference_latents is not None else images, *args, **kwargs
        )
        if self.reference_latents is not None:
            image_latents = self.reference_latents.to(device=latents.device, dtype=latents.dtype)
        return latents, image_latents


#: Runtime code the encoders read beyond their own bodies: their memo key binds it.
_ENCODER_CODE = (_StagedPipeline, build_processor)


class QwenImage21Model(Model[QwenImage21Graph]):
    graph: QwenImage21Graph
    processor: Any

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(QwenImage21Graph, factory=build_qwen_image21)
        self.processor = build_processor(loader.assets)

    def execution_plan(self, plan: Plan) -> str:
        assert BLOCK is not None
        return fusion.install(plan, BLOCK, self.graph.components["transformer"])

    def _pipeline(self) -> _StagedPipeline:
        return _StagedPipeline(
            **self.graph.components,
            processor=self.processor,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.graph.scheduler_config),
        )

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode(self, prompt: str) -> tuple[Any, Any]:
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            embeds, mask, _ = self._pipeline().encode_prompt(
                prompt, device=next(encoder.parameters()).device
            )
        return embeds, mask

    @uses_components("vae", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_reference_images(self, images: list[Any]) -> tuple[list[Any], torch.Tensor]:
        """Encode each reference once, using Diffusers' 1 MP conditioning size."""
        if not 1 <= len(images) <= 10:
            raise ValueError("Qwen Image requires between one and ten reference images")
        vae = self.graph.components["vae"]
        device = next(vae.parameters()).device
        pipeline = self._pipeline()
        resized, packed = [], []
        with torch.inference_mode():
            for image in images:
                image = image.convert("RGBA")
                width, height, _ = calculate_dimensions(1024 * 1024, image.width / image.height)
                resized.append(pipeline.image_processor.resize(image, width=width, height=height))
                pixels = (
                    pipeline.image_processor.preprocess(image, width=width, height=height)
                    .unsqueeze(2)
                    .to(device=device, dtype=vae.dtype)
                )
                # argmax encoding is deterministic and consumes no sampler RNG.
                encoded = pipeline._encode_vae_image(pixels, generator=None)
                packed.append(
                    pipeline._pack_latents(
                        encoded, 1, pipeline.latent_channels, encoded.shape[3], encoded.shape[4]
                    ).cpu()
                )
        return resized, torch.cat(packed, dim=1)

    @uses_components("text_encoder", memoize=True, memo_dependencies=_ENCODER_CODE)
    def encode_image_prompt(self, prompt: str, images: list[Any]) -> tuple[Any, Any, Any]:
        """Encode the instruction and ordered reference images as one condition."""
        encoder = self.graph.components["text_encoder"]
        with torch.inference_mode():
            encoded: tuple[Any, Any, Any] = self._pipeline().encode_prompt(
                prompt, image=images, device=next(encoder.parameters()).device
            )
        return encoded

    @uses_components("transformer")
    def denoise(
        self,
        embeds: Any,
        mask: Any,
        *,
        width: int,
        height: int,
        steps: int,
        seed: int,
        on_step: Callable[[int], None],
        cancel: Callable[[], None],
        reference_images: list[Any] | None = None,
        reference_latents: torch.Tensor | None = None,
        image_pad_mask: torch.Tensor | None = None,
    ) -> Any:
        transformer = self.graph.components["transformer"]
        device = next(transformer.parameters()).device
        pipeline = self._pipeline()
        if reference_images is not None:
            if not reference_images or reference_latents is None or image_pad_mask is None:
                raise ValueError(
                    "Qwen Image references require images, latents and vision token mask"
                )
            pipeline.reference_latents = reference_latents
            pipeline.reference_pad_mask = image_pad_mask.to(device=device)
        elif reference_latents is not None or image_pad_mask is not None:
            raise ValueError("Qwen Image reference tensors require their source images")

        def observed(_pipeline: Any, step: int, _timestep: Any, values: Any) -> Any:
            cancel()
            on_step(step)
            return values

        cancel()
        with torch.inference_mode():
            return pipeline(
                image=reference_images,
                prompt_embeds=embeds.to(device=device, dtype=transformer.dtype),
                prompt_embeds_mask=None if mask is None else mask.to(device=device),
                width=width,
                height=height,
                num_inference_steps=steps,
                generator=torch.Generator(device=device).manual_seed(seed),
                output_type="latent",
                callback_on_step_end=observed,
            ).images

    @uses_components("vae")
    def decode(self, latents: Any, *, width: int, height: int) -> Any:
        vae = self.graph.components["vae"]
        device = next(vae.parameters()).device
        with torch.inference_mode():
            latents = QwenImage21Pipeline._unpack_latents(latents, height, width, 16)
            latents = latents.to(device=device, dtype=vae.dtype)
            mean = latents.new_tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
            std = latents.new_tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
            # The pipeline's final decode, with no cast of F32 source weights, as one frame:
            # Diffusers' `decode` keeps every convolution's input for a next frame an image
            # never has (its feature cache, half the decode's memory); the same bits without.
            x = vae.post_quant_conv(latents * std + mean)
            return vae.decoder(x, first_chunk=True).clamp(-1.0, 1.0)[:, :, 0]
