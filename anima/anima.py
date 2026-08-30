"""Anima text-to-image through Diffusers' maintained modular pipeline."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import msgspec
from cozy_runtime.author import (
    App,
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


_BUCKETS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1024, 1024),
    AspectRatio.LANDSCAPE: (1152, 896),
    AspectRatio.WIDE: (1344, 768),
    AspectRatio.PORTRAIT: (896, 1152),
    AspectRatio.TALL: (768, 1344),
}


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "masterpiece, best quality, 1girl, solo, city lights"
    negative_prompt: str = "low quality, worst quality, blurry"
    aspect_ratio: Annotated[AspectRatio, Shape(pixels=_BUCKETS)] = AspectRatio.SQUARE
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=8, le=50)] = 30
    guidance: Annotated[ModelDefault[float], msgspec.Meta(ge=1.0, le=10.0)] = 4.5
    seed: int = 1005


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    width: int
    height: int
    steps: int
    guidance: float
    digest: str


def _tokenizer(path: Path) -> Any:
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=str(path / "tokenizer.json"))


class AnimaPipeline:
    def __init__(self, config: Any) -> None:
        import torch
        from diffusers import (
            AnimaTextConditioner,
            AutoencoderKLQwenImage,
            CosmosTransformer3DModel,
        )
        from transformers import Qwen3Config, Qwen3Model

        mapping = config.mapping()
        transformer = CosmosTransformer3DModel.from_config(mapping["transformer"]).to(
            torch.bfloat16
        )
        text_encoder: Any = Qwen3Model(Qwen3Config(**mapping["text_encoder"]))
        text_encoder.to(dtype=torch.bfloat16)
        text_conditioner = AnimaTextConditioner.from_config(mapping["text_conditioner"]).to(
            torch.bfloat16
        )
        vae = AutoencoderKLQwenImage.from_config(mapping["vae"]).to(torch.bfloat16)
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
    ) -> Any:
        import torch
        from diffusers import AnimaModularPipeline, FlowMatchEulerDiscreteScheduler

        device = next(self.pipe.components["transformer"].parameters()).device
        generator = torch.Generator(device=device).manual_seed(seed)
        pipeline: Any = AnimaModularPipeline(workflow="text2image")
        pipeline.register_components(
            **self.pipe.components,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(self.pipe.scheduler_config),
            tokenizer=self.pipe.tokenizer,
            t5_tokenizer=self.pipe.t5_tokenizer,
        )
        pipeline.set_progress_bar_config(disable=True)
        return pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            output="images",
            output_type="pt",
        )


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

    width, height = _BUCKETS[payload.aspect_ratio]
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
        )
    image = images[0]
    pixels = (image.clamp(0, 1) * 255).to("cpu", dtype=torch.uint8)
    if pixels.ndim == 3 and pixels.shape[0] == 3:
        pixels = pixels.permute(1, 2, 0)
    pixels = pixels.contiguous()
    rgb = bytes(pixels.numpy().tobytes())
    asset = out.save_image(ImageFrame(width, height, rgb), format="png")
    return ImageOutput(
        asset, width, height, steps, payload.guidance, hashlib.sha256(rgb).hexdigest()
    )
