"""Private matched H3 latent-upscale experiment; never a public H3 release API."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import sys
import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from typing import Annotated, Literal, Protocol, TypedDict, cast

import msgspec
import cozy_runtime
from cozy_runtime import _build_provenance
import tensorfs
import torch
from cozy_runtime.author import (
    App,
    Assets,
    AssetBound,
    Context,
    FileAsset,
    ImageAsset,
    Loader,
    Model,
    ModelArtifact,
    Outputs,
    PendingCall,
    Shape,
    Telemetry,
    VideoAsset,
    WeightsOutput,
    invocable,
    uses_components,
)
from cozy_runtime.author import Config as ModelConfig
from cozy_runtime.author.sources import download_huggingface, convert_cozytensors
from cozy_runtime.models.minimax_h3.continuation import plan_continuation
from cozy_runtime.models.minimax_h3.model import H3TurboBase, H3TurboLoRA, condition_references
from cozy_runtime.models.minimax_h3.official import (
    NumericalChecks,
    ScheduleFacts,
    OfficialH3Pipeline,
)
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors
from tensorfs.derived import Config, Derivation, Part, Target, Tensor

from h3 import (
    ReferenceAssets,
    TURBO_STEPS,
    _finish,
    assets_to_h3_refs,
)
from latent_upscale_net import LATENTS_MEAN, LATENTS_STD, LatentResizer3D
from story import encoder_prompt

app = App()

CANVASES = {"preview": (960, 480), "native": (1536, 768)}
# The requested pair uses one identical H3 gang on the verified Asirpa GPU SKU.
BENCH_MODEL: list[dict[str, str | int]] = [
    {"gpu": "H100", "gpus": 4, "lane": "minimax-h3@1.0.0-rc.2/fp8-pruned"}
]
BENCH_LORA: list[dict[str, str | int]] = [
    {"gpu": "H100", "gpus": 4, "lane": "minimax-h3-turbo-lora@1.0.0-audit.1/pdd8"}
]
Canvas = Annotated[Literal["preview", "native"], Shape(pixels=CANVASES)]
Duration = Annotated[Literal[10], Shape(frames={10: 243})]
LatentFile = Annotated[FileAsset, AssetBound(max_bytes=128 << 20)]


class UpscaleConfig(TypedDict):
    in_channels: int
    in_blocks: int
    out_blocks: int
    channels: int
    dropout: float
    temporal_every: int
    temporal_kernel: int


WEIGHT_CONFIG: UpscaleConfig = {
    "in_channels": 24,
    "in_blocks": 12,
    "out_blocks": 12,
    "channels": 512,
    "dropout": 0.1,
    "temporal_every": 2,
    "temporal_kernel": 5,
}
WEIGHT_SHA256 = "043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2"
WEIGHT_SOURCE = (
    "https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler/resolve/"
    "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/"
    "minimax_h3_latent_upscaler_3d_conv_v1/"
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
)


class StageTimes(msgspec.Struct):
    prepare: float = 0.0
    condition: float = 0.0
    denoise: float = 0.0
    retain_latents: float = 0.0
    upscale: float = 0.0


@contextmanager
def stage(tel: Telemetry, clocks: StageTimes, name: str) -> Iterator[None]:
    """Completed CPU reads synchronize CUDA results before each measured scope exits."""
    start = time.perf_counter()
    with tel.stage(name):
        yield
    setattr(clocks, name, time.perf_counter() - start)


def digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().view(torch.uint8).numpy()).hexdigest()


class ScheduleInfo(msgspec.Struct):
    timestep_plan_digest: str
    transformer_evaluations: int
    sigma_grid_points: int
    video_sigma_digest: str
    audio_sigma_digest: str
    video_timestep_digest: str
    audio_timestep_digest: str


class LatentInfo(msgspec.Struct, kw_only=True):
    schema: Literal["h3-preview-latents/1"]
    width: int
    height: int
    duration_s: int
    sampled_frames: int
    delivered_frames: int
    seed: int
    torch_seed: int
    model_manifest: str
    turbo_lora_manifest: str
    prompt: str
    reference_digests: list[str]
    reference_sizes: list[list[int]]
    video_shape: list[int]
    audio_shape: list[int]
    video_digest: str
    audio_digest: str
    schedule: ScheduleInfo
    stages_s: StageTimes
    source_video_digest: str | None = None
    upscale_scale: float | None = None
    upscale_weights: str | None = None


class Latents(msgspec.Struct):
    file: LatentFile
    info: LatentInfo


def retain(out: Outputs, video: torch.Tensor, audio: torch.Tensor, info: LatentInfo) -> Latents:
    tensors = {
        "video": video.detach().contiguous().cpu(),
        "audio": audio.detach().contiguous().cpu(),
    }
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("completed latents contain non-finite values")
    info.video_shape = list(tensors["video"].shape)
    info.audio_shape = list(tensors["audio"].shape)
    info.video_digest = digest(tensors["video"])
    info.audio_digest = digest(tensors["audio"])
    return Latents(
        out.save_bytes(save_tensors(tensors), media_type="application/octet-stream"), info
    )


def restore(value: Latents) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = load_tensors(value.file.read_bytes())
    if set(tensors) != {"video", "audio"}:
        raise ValueError("latent file has an unexpected tensor set")
    video, audio = tensors["video"], tensors["audio"]
    info = value.info
    if list(video.shape) != info.video_shape or list(audio.shape) != info.audio_shape:
        raise ValueError("latent geometry differs from its receipt")
    if video.ndim != 5 or tuple(video.shape[:2]) != (1, 24):
        raise ValueError("video latent shape must be BCTHW with 24 channels")
    if tuple(video.shape[-2:]) != (info.height // 16, info.width // 16):
        raise ValueError("video latent canvas differs from its receipt")
    if audio.ndim != 3 or tuple(audio.shape[:2]) != (2, 32):
        raise ValueError("audio latent shape must be stereo, 32 channels")
    if digest(video) != info.video_digest or digest(audio) != info.audio_digest:
        raise ValueError("latent content differs from its receipt")
    return video, audio


class GenerateInput(msgspec.Struct, kw_only=True):
    prompt: str
    canvas: Canvas
    seed: int
    duration_s: Duration = 10


@invocable(defaults={"base_model": BENCH_MODEL, "turbo_lora": BENCH_LORA})
async def generate_latents(
    ctx: Context,
    *,
    payload: GenerateInput,
    assets: ReferenceAssets,
    base_model: H3TurboBase,
    turbo_lora: H3TurboLoRA,
    out: Outputs,
    tel: Telemetry,
) -> Latents:
    ctx.raise_if_cancelled()
    clocks = StageTimes()
    width, height = CANVASES[payload.canvas]
    delivery = plan_continuation(payload.duration_s * 24)
    view = base_model.for_request(ctx, seed=payload.seed)
    generator = base_model.pipe.generator(view.generator)
    checks = NumericalChecks(tel, base_model.pipe.resident)
    with stage(tel, clocks, "prepare"):
        references, sizing = assets_to_h3_refs(assets, pipe=base_model.pipe)
        prompt = encoder_prompt("ref2va_turbo", payload.prompt)
        state = base_model.pipe.start_ref2va(
            prompt=prompt,
            references=references,
            generator=generator,
            steps=TURBO_STEPS,
            frames=delivery.sample_frames,
            delivery=delivery,
            reference_image_short_edges=sizing.edges,
            task="ref2va_turbo",
            width=width,
            height=height,
        )
        if (state.width, state.height) != (width, height):
            raise ValueError("upstream changed the explicit benchmark canvas")
        reference_sizes = [list(ref.image.size) for ref in state.normalized_references]
    with stage(tel, clocks, "condition"):
        condition_references(base_model, "ref2va_turbo", state, checks=checks)
    with stage(tel, clocks, "denoise"):
        schedule = base_model.sample_ref2va_turbo(
            state,
            turbo_lora=turbo_lora,
            on_step=tel.step_callback(TURBO_STEPS, stage="denoise"),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
        # The completed readback fences denoise; disk serialization has a separate clock.
        video = state.latents.detach().contiguous().cpu()
        audio = state.audio_latents.detach().contiguous().cpu()
    info = LatentInfo(
        schema="h3-preview-latents/1",
        width=width,
        height=height,
        duration_s=payload.duration_s,
        sampled_frames=delivery.sample_frames,
        delivered_frames=delivery.delivered_frames,
        seed=payload.seed,
        torch_seed=generator.initial_seed(),
        model_manifest=base_model.checkpoint_ref,
        turbo_lora_manifest=turbo_lora.checkpoint_ref,
        prompt=prompt,
        reference_digests=[assets.info(i).digest for i in range(len(assets))],
        reference_sizes=reference_sizes,
        video_shape=[],
        audio_shape=[],
        video_digest="",
        audio_digest="",
        schedule=ScheduleInfo(**asdict(schedule)),
        stages_s=clocks,
    )
    with stage(tel, clocks, "retain_latents"):
        result = retain(out, video, audio, info)
    tel.log("matched benchmark generated latents", receipt=json.dumps(msgspec.to_builtins(info)))
    return result


class UpscalePipeline:
    def __init__(self, configuration: UpscaleConfig) -> None:
        self.components = {"upscaler": LatentResizer3D(**configuration).eval()}


def build_upscaler(config: ModelConfig) -> UpscalePipeline:
    configuration: UpscaleConfig = {
        "in_channels": config.as_int("in_channels"),
        "in_blocks": config.as_int("in_blocks"),
        "out_blocks": config.as_int("out_blocks"),
        "channels": config.as_int("channels"),
        "dropout": config.as_float("dropout"),
        "temporal_every": config.as_int("temporal_every"),
        "temporal_kernel": config.as_int("temporal_kernel"),
    }
    if configuration != WEIGHT_CONFIG:
        raise ValueError("unsupported benchmark upscaler architecture")
    return UpscalePipeline(configuration)


class LatentUpscaler(Model[UpscalePipeline]):
    pipe: UpscalePipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(UpscalePipeline, factory=build_upscaler)

    @uses_components("upscaler")
    def upscale(self, video: torch.Tensor, width: int, height: int) -> torch.Tensor:
        net = self.pipe.components["upscaler"]
        anchor = next(net.parameters())
        if anchor.dtype != torch.float32:
            raise ValueError("Director benchmark requires FP32 upscaler inference")
        if width * video.shape[-2] != height * video.shape[-1]:
            raise ValueError("upscale must preserve aspect ratio")
        scale = width / (16 * video.shape[-1])
        target = (video.shape[2], height // 16, width // 16)
        with torch.inference_mode():
            x = video.to(device=anchor.device, dtype=torch.float32)
            mean = x.new_tensor(LATENTS_MEAN).view(1, -1, 1, 1, 1)
            std = x.new_tensor(LATENTS_STD).view(1, -1, 1, 1, 1)
            y = net((x - mean) / std, scale, target, enable_chunking=False)
            if not isinstance(y, torch.Tensor):
                raise ValueError("upscaler did not return a tensor")
            return (y * std + mean).to(dtype=video.dtype).cpu()


class UpscalerSource(Model[object]):
    def load(self, loader: Loader) -> None:
        del loader


@invocable(memoize=True)
async def prepare_upscaler(
    ctx: Context, *, source: UpscalerSource, tel: Telemetry
) -> ModelArtifact:
    """Validate native source parts and derive FP32 weights with the normal writer."""
    with torch.device("meta"):
        expected = LatentResizer3D(**WEIGHT_CONFIG).state_dict()
    plain = dict(tensorfs.seed_digests())["plain/1"]
    with ctx.tensorfs_source(source) as native:
        structure = native.inspect()
        if set(structure.components) != {"model"}:
            raise ValueError("upscaler source must contain its one as-is model component")
        tensors = structure.components["model"]
        if set(tensors) != set(expected):
            raise ValueError("upscaler source tensor names differ from the reviewed architecture")
        for key, item in tensors.items():
            if (
                tuple(item.shape) != tuple(expected[key].shape)
                or item.logical_dtype != "f16"
                or item.encoding != plain
                or set(item.parts) != {"value"}
                or item.parts["value"].dtype != "f16"
                or tuple(item.parts["value"].shape) != tuple(expected[key].shape)
            ):
                raise ValueError(f"upscaler source tensor {key} differs from the reviewed layout")
        order = tuple(sorted(tensors))
        definition = Derivation(
            sources={},
            targets={
                "upscaler": Target(
                    add={
                        key: Tensor(
                            "f32",
                            tuple(expected[key].shape),
                            plain,
                            {"value": Part("f32", tuple(expected[key].shape))},
                        )
                        for key in order
                    }
                )
            },
            configs={"pipeline": Config("add")},
            order=tuple(("upscaler", key) for key in order),
        )
        with ctx.output("upscaler").open(definition) as transaction:
            if transaction.receipt is None:
                progress = tel.step_callback(len(order), stage="convert_upscaler")
                for index, key in enumerate(order):
                    ctx.raise_if_cancelled()
                    value = torch.empty(tuple(expected[key].shape), dtype=torch.float16)
                    native.read_part_into(
                        "model",
                        key,
                        "value",
                        0,
                        memoryview(value.view(torch.uint8).numpy()).cast("B"),
                    )
                    converted = value.to(torch.float32)
                    if not bool(torch.isfinite(converted).all()):
                        raise ValueError(f"upscaler weight {key} is non-finite")
                    transaction.add_part("upscaler", key, "value", converted.numpy().tobytes())
                    progress(index)
                transaction.add_config("pipeline", json.dumps(WEIGHT_CONFIG).encode())
            return ctx.adopt_model(transaction.receipt or transaction.commit())


app.job(
    prepare_upscaler, weights=(WeightsOutput("upscaler", max_new_bytes=2 << 30),), accelerator=False
)


class UpscaleInput(msgspec.Struct):
    latents: Latents
    canvas: Annotated[Literal["native"], Shape(pixels={"native": (1536, 768)})] = "native"
    duration_s: Duration = 10


@invocable
async def upscale_latents(
    ctx: Context,
    *,
    payload: UpscaleInput,
    model: LatentUpscaler,
    out: Outputs,
    tel: Telemetry,
) -> Latents:
    video, audio = restore(payload.latents)
    info = msgspec.structs.replace(
        payload.latents.info, stages_s=msgspec.structs.replace(payload.latents.info.stages_s)
    )
    width, height = CANVASES[payload.canvas]
    ctx.raise_if_cancelled()
    with stage(tel, info.stages_s, "upscale"):
        lifted = model.upscale(video, width, height)
    if lifted.shape[:3] != video.shape[:3]:
        raise ValueError("spatial upscale changed the temporal grid")
    info.source_video_digest = info.video_digest
    info.upscale_scale = width / info.width
    info.upscale_weights = model.checkpoint_ref
    info.width, info.height = width, height
    result = retain(out, lifted, audio, info)
    if result.info.audio_digest != payload.latents.info.audio_digest:
        raise ValueError("upscale changed audio")
    return result


class DecodeInput(msgspec.Struct):
    latents: Latents
    canvas: Annotated[Literal["native"], Shape(pixels={"native": (1536, 768)})] = "native"
    duration_s: Duration = 10


class Rendered(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    info: LatentInfo
    decode_and_encode_s: float
    warnings: list[str]


@invocable(defaults={"base_model": BENCH_MODEL})
async def decode_latents(
    ctx: Context,
    *,
    payload: DecodeInput,
    base_model: H3TurboBase,
    out: Outputs,
    tel: Telemetry,
) -> Rendered:
    video, audio = restore(payload.latents)
    info = payload.latents.info
    if (info.width, info.height) != CANVASES[payload.canvas]:
        raise ValueError("decode canvas differs from measured admission shape")
    state = PipelineState()
    state.set("latents", video)
    state.set("audio_latents", audio)
    state.set("output_type", "pt")
    state.set("continuation_delivery", plan_continuation(info.delivered_frames))
    started = time.perf_counter()
    rendered = _finish(
        base_model,
        "ref2va_turbo",
        state,
        ScheduleFacts(**msgspec.to_builtins(info.schedule)),
        duration_s=info.duration_s,
        out=out,
        tel=tel,
        cancel=ctx.raise_if_cancelled,
        checks=NumericalChecks(tel, base_model.pipe.resident),
        delivered_frames=info.delivered_frames,
        sampled_frames=info.sampled_frames,
    )
    return Rendered(rendered.video, info, time.perf_counter() - started, rendered.warnings)


class PairInput(msgspec.Struct, kw_only=True):
    prompt: str
    seed: int
    reference_images: list[ImageAsset]
    duration_s: Duration = 10


class PairOutput(msgspec.Struct):
    preview_upscaled: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    native: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    metadata: Annotated[FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))]


# Managed call sites expose the static wire signature, while the registered
# implementation also accepts injected services/models. The broker proof checks
# these exact contracts through the real generated request/result codecs.
class PrepareCall(Protocol):
    def __call__(self, *, source: ModelArtifact) -> PendingCall[ModelArtifact]: ...


class GenerateCall(Protocol):
    def __call__(
        self, *, payload: GenerateInput, assets: Assets[ImageAsset]
    ) -> PendingCall[Latents]: ...


class UpscaleCall(Protocol):
    def __call__(self, *, payload: UpscaleInput, model: ModelArtifact) -> PendingCall[Latents]: ...


class DecodeCall(Protocol):
    def __call__(self, *, payload: DecodeInput) -> PendingCall[Rendered]: ...


async def compare(ctx: Context, *, payload: PairInput, out: Outputs, tel: Telemetry) -> PairOutput:
    if len(payload.reference_images) != 2:
        raise ValueError("benchmark requires the approved character and background images")
    ctx.raise_if_cancelled()
    start = time.perf_counter()
    checkpoint_start = time.perf_counter()
    source = await download_huggingface(
        "LBH-123-AI/Minimax_h3_latent_Upscaler",
        revision="3f941d5d182014dd5c0a5e16330420ee2d4aa0c6",
        carriers=(
            "minimax_h3_latent_upscaler_3d_conv_v1/"
            "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
        ),
    )
    native_source = await convert_cozytensors(source, profiles=("as-is/1",))
    checkpoint = await cast(PrepareCall, prepare_upscaler)(source=native_source)
    conversion_wall = time.perf_counter() - checkpoint_start
    a_start = time.perf_counter()
    a_call = cast(GenerateCall, generate_latents)(
        payload=GenerateInput(
            prompt=payload.prompt,
            canvas="preview",
            seed=payload.seed,
            duration_s=payload.duration_s,
        ),
        assets=Assets(payload.reference_images),
    )
    a_raw = await a_call
    upscale_call = cast(UpscaleCall, upscale_latents)(payload=UpscaleInput(a_raw), model=checkpoint)
    a_lifted = await upscale_call
    a_decode = cast(DecodeCall, decode_latents)(payload=DecodeInput(a_lifted))
    a = await a_decode
    a_wall = time.perf_counter() - a_start
    b_start = time.perf_counter()
    b_call = cast(GenerateCall, generate_latents)(
        payload=GenerateInput(
            prompt=payload.prompt, canvas="native", seed=payload.seed, duration_s=payload.duration_s
        ),
        assets=Assets(payload.reference_images),
    )
    b_raw = await b_call
    b_decode = cast(DecodeCall, decode_latents)(payload=DecodeInput(b_raw))
    b = await b_decode
    b_wall = time.perf_counter() - b_start
    if a.info.prompt != b.info.prompt or a.info.reference_sizes != b.info.reference_sizes:
        raise ValueError("paired conditioning differs")
    if a.info.reference_digests != b.info.reference_digests or a.info.schedule != b.info.schedule:
        raise ValueError("paired references or sampler differs")
    if a.info.seed != b.info.seed or a.info.torch_seed != b.info.torch_seed:
        raise ValueError("paired numeric seeds differ")
    if (a.info.model_manifest, a.info.turbo_lora_manifest) != (
        b.info.model_manifest,
        b.info.turbo_lora_manifest,
    ):
        raise ValueError("paired admitted model manifests differ")
    observed_calls: list[tuple[str, PendingCall[Latents] | PendingCall[Rendered]]] = [
        ("A_generate", a_call),
        ("A_upscale", upscale_call),
        ("A_decode", a_decode),
        ("B_generate", b_call),
        ("B_decode", b_decode),
    ]
    metadata = {
        "schema": "cozy.h3-preview-comparison/1",
        "request_id": ctx.request_id,
        "order": ["preview_upscaled", "native"],
        "total_wall_s": time.perf_counter() - start,
        "upscaler_conversion_wall_s": conversion_wall,
        "upscaler_source": WEIGHT_SOURCE,
        "upscaler_weight_digest": WEIGHT_SHA256,
        "upscaler_native_source": msgspec.to_builtins(native_source),
        "upscaler_checkpoint": msgspec.to_builtins(checkpoint),
        "A": {
            "latents": msgspec.to_builtins(a.info),
            "decode_and_encode_s": a.decode_and_encode_s,
            "wall_s": a_wall,
            "video_digest": a.video.digest,
            "warnings": a.warnings,
        },
        "B": {
            "latents": msgspec.to_builtins(b.info),
            "decode_and_encode_s": b.decode_and_encode_s,
            "wall_s": b_wall,
            "video_digest": b.video.digest,
            "warnings": b.warnings,
        },
        "child_execution": {
            name: {
                "request_id": call.request_id,
                "observation": msgspec.to_builtins(call.observation),
            }
            for name, call in observed_calls
        },
        "limits": [
            "Prepare/condition are host intervals; asynchronous GPU work can finish in the fenced denoise readback.",
            "Actual GPU UUIDs and per-GPU attention choices are read from these child IDs in Runtime execution evidence.",
            "One ordered pair: first-arm cold preparation is not a fair warm-speed comparison.",
            "Same numeric seed; resolution changes noise grids and subsequent audio RNG draws.",
            "Both benchmark canvases are 2:1; source run 1937 was 1344x768.",
            "A uses learned spatial lift without re-denoising; fine details can change.",
        ],
    }
    tel.log("matched benchmark complete", a_wall_s=a_wall, b_wall_s=b_wall)
    return PairOutput(
        a.video,
        b.video,
        out.save_bytes(json.dumps(metadata, indent=2).encode(), media_type="application/json"),
    )


app.entrypoint(internal=True)(generate_latents)
app.entrypoint(internal=True)(upscale_latents)
app.entrypoint(internal=True)(decode_latents)
app.job(compare, emits_media=True, accelerator=False)


class Software(msgspec.Struct):
    python: str
    runtime: str
    runtime_module: str
    source_wheel_sha256: str
    tensorfs: str
    package: str
    explicit_canvas: bool
    cuda_initialized: bool


class SoftwareInput(msgspec.Struct):
    pass


def software(ctx: Context, payload: SoftwareInput) -> Software:
    """CPU-only readback of the executor's actual selected experimental SDK."""
    del payload
    ctx.raise_if_cancelled()
    parameters = inspect.signature(OfficialH3Pipeline.start_ref2va).parameters
    return Software(
        sys.version,
        importlib.metadata.version("cozy-runtime"),
        str(cozy_runtime.__file__),
        getattr(_build_provenance, "SOURCE_WHEEL_SHA256", ""),
        importlib.metadata.version("tensorfs"),
        importlib.metadata.version("h3-preview-ab"),
        "height" in parameters and "width" in parameters,
        cast(Callable[[], bool], torch.cuda.is_initialized)(),
    )


app.job(software, accelerator=False)
