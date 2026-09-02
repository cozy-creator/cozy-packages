"""Anima text-to-image through Diffusers' maintained modular pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from enum import Enum, IntEnum
from pathlib import Path
from typing import Annotated, Any

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    Loader,
    Model,
    ModelDefault,
    Outputs,
    Shape,
    Telemetry,
    uses_components,
)

app = App()
_MODULE_ROOT = Path(__file__).resolve().parent
_ROOT = _MODULE_ROOT / "anima_assets" if (_MODULE_ROOT / "anima_assets").is_dir() else _MODULE_ROOT


class AspectRatio(Enum):
    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    WIDE = "16:9"
    PORTRAIT = "3:4"
    TALL = "9:16"


class Megapixels(IntEnum):
    """Anima's native resolution classes, in nominal megapixels (se-024).

    Tier 2 is the 1536-class (~2.3 MP) the model is meant to be run at and this
    package's default; tier 1 trades resolution for speed. Denoise attention is full
    (no windowing), so time and VRAM scale roughly linearly with area — tier 1 costs
    about half of tier 2. The VAE always tiles. Every bucket is a multiple of the
    pipeline's 16-px stride.
    """

    MP1 = 1
    MP2 = 2


#: (aspect, tier) -> (width, height): tier 1 is the ~1 MP training set, tier 2 scales it
#: by exactly 1.5 to the 1536 class. A pair absent here does not round — it refuses at
#: decode (today the grid is complete, so only an out-of-enum value can refuse).
_BUCKETS: dict[tuple[AspectRatio, Megapixels], tuple[int, int]] = {
    (AspectRatio.SQUARE, Megapixels.MP1): (1024, 1024),
    (AspectRatio.LANDSCAPE, Megapixels.MP1): (1152, 896),
    (AspectRatio.WIDE, Megapixels.MP1): (1344, 768),
    (AspectRatio.PORTRAIT, Megapixels.MP1): (896, 1152),
    (AspectRatio.TALL, Megapixels.MP1): (768, 1344),
    (AspectRatio.SQUARE, Megapixels.MP2): (1536, 1536),
    (AspectRatio.LANDSCAPE, Megapixels.MP2): (1728, 1344),
    (AspectRatio.WIDE, Megapixels.MP2): (2016, 1152),
    (AspectRatio.PORTRAIT, Megapixels.MP2): (1344, 1728),
    (AspectRatio.TALL, Megapixels.MP2): (1152, 2016),
}

#: The demand table `Shape` reads: the runtime derives (width, height, pixels) from ONE
#: field, so the tier carries its largest bucket as an upper bound over the tier.
_TIER_DEMAND: dict[Megapixels, tuple[int, int]] = {
    tier: max(
        (size for (_, t), size in _BUCKETS.items() if t is tier),
        key=lambda size: size[0] * size[1],
    )
    for tier in Megapixels
}
_WEBP_OUTPUT = AssetBound(max_bytes=64 << 20, media_types=("image/webp",))


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "masterpiece, best quality, 1girl, solo, city lights"
    negative_prompt: str = "low quality, worst quality, blurry"
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    megapixels: Annotated[Megapixels, Shape(pixels=_TIER_DEMAND)] = Megapixels.MP2
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=8, le=50)] = 30
    guidance: Annotated[ModelDefault[float], msgspec.Meta(ge=1.0, le=10.0)] = 4.5
    seed: int = 1005


class ImageOutput(msgspec.Struct):
    image: Annotated[ImageAsset, _WEBP_OUTPUT]
    width: int
    height: int
    steps: int
    guidance: float
    digest: str


def _tokenizer(path: Path) -> Any:
    from transformers import PreTrainedTokenizerFast

    config = json.loads((path / "tokenizer_config.json").read_text())
    config.pop("tokenizer_class", None)
    return PreTrainedTokenizerFast(
        tokenizer_file=str(path / "tokenizer.json"), **config
    )


class AnimaPipeline:
    def __init__(self, config: Any) -> None:
        import torch
        from diffusers import (
            AnimaTextConditioner,
            AutoencoderKLQwenImage,
            CosmosTransformer3DModel,
        )
        from transformers import Qwen3Config, Qwen3Model
        from transformers import initialization as transformer_init

        mapping = config.mapping()
        with transformer_init.no_init_weights():
            transformer = CosmosTransformer3DModel.from_config(mapping["transformer"]).to(
                torch.bfloat16
            )
            text_encoder: Any = Qwen3Model(Qwen3Config(**mapping["text_encoder"]))
            text_encoder.to(dtype=torch.bfloat16)
            text_conditioner = AnimaTextConditioner.from_config(
                mapping["text_conditioner"]
            ).to(torch.bfloat16)
            vae = AutoencoderKLQwenImage.from_config(mapping["vae"]).to(torch.bfloat16)

        for component in (transformer, text_encoder, text_conditioner, vae):
            component.eval()
        vae.enable_tiling()
        self.scheduler_config = mapping["scheduler"]
        self.tokenizer = _tokenizer(_ROOT / "tokenizer")
        self.t5_tokenizer = _tokenizer(_ROOT / "t5_tokenizer")
        self.components: dict[str, Any] = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "text_conditioner": text_conditioner,
            "vae": vae,
        }


def build_pipeline(config: Any) -> AnimaPipeline:
    return AnimaPipeline(config)


class AnimaModel(Model[AnimaPipeline]):
    pipe: AnimaPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(AnimaPipeline, factory=build_pipeline)

    @uses_components("text_encoder", "text_conditioner", "transformer", "vae")
    def render(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        seed: int,
        tel: Telemetry,
    ) -> Any:
        import torch
        from diffusers import AnimaModularPipeline, FlowMatchEulerDiscreteScheduler

        device = next(self.pipe.components["transformer"].parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)

        class RuntimeAnimaPipeline(AnimaModularPipeline):
            @property
            def _execution_device(self) -> Any:
                return device

        pipeline: Any = RuntimeAnimaPipeline(workflow="text2image")
        pipeline.register_components(
            **self.pipe.components,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler_config),
            tokenizer=self.pipe.tokenizer,
            t5_tokenizer=self.pipe.t5_tokenizer,
        )
        pipeline.guider.guidance_scale = guidance
        denoise = pipeline.blocks.sub_blocks.get("denoise.denoise")
        if denoise is None:
            raise RuntimeError("Diffusers Anima workflow has no denoise.denoise block")
        denoise.progress_bar = _progress_bar(tel)
        tel.progress(0, stage="conditioning")
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
            output="images",
            output_type="pt",
        )


class _DenoiseProgress:
    """Diffusers' denoise-loop progress bar projected onto Runtime telemetry."""

    def __init__(self, total: int, tel: Telemetry) -> None:
        self.total = total
        self.tel = tel
        self.position = 0
        self.step: Callable[[int], None] | None = None

    def __enter__(self) -> _DenoiseProgress:
        self.step = self.tel.step_callback(self.total, stage="denoise")
        return self

    def update(self, count: int = 1) -> None:
        if self.step is None:
            raise RuntimeError("Anima denoise progress updated outside its loop")
        for _ in range(count):
            self.step(self.position)
            self.position += 1

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.tel.progress(0, stage="decoding")


def _progress_bar(tel: Telemetry) -> Callable[..., _DenoiseProgress]:
    def progress_bar(iterable: object = None, total: int | None = None) -> _DenoiseProgress:
        if iterable is not None or total is None or total < 1:
            raise RuntimeError("Anima denoise progress requires one positive total")
        return _DenoiseProgress(total, tel)

    return progress_bar


@app.entrypoint
def generate(
    ctx: Context,
    payload: GenerateInput,
    model: AnimaModel,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    """Generate one native-resolution Anima image."""
    import torch

    width, height = _BUCKETS[(payload.aspect_ratio, payload.megapixels)]
    steps = payload.steps
    if ctx.boot_warmup:
        width = height = 512
        steps = 1
    with tel.stage("generate"):
        images = model.render(
            payload.prompt,
            payload.negative_prompt,
            width,
            height,
            steps,
            payload.guidance,
            payload.seed,
            tel,
        )
    image = images[0]
    pixels = (image.clamp(0, 1) * 255).to("cpu", dtype=torch.uint8)
    if pixels.ndim == 3 and pixels.shape[0] == 3:
        pixels = pixels.permute(1, 2, 0)
    pixels = pixels.contiguous()
    rgb = bytes(pixels.numpy().tobytes())
    with tel.stage("encode_webp"):
        asset = out.save_image(ImageFrame(width, height, rgb), format="webp")
    return ImageOutput(
        asset, width, height, steps, payload.guidance, hashlib.sha256(rgb).hexdigest()
    )
