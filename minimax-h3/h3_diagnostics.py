"""Optional H3 observations over the unchanged inference dependency and Runtime scopes."""

import importlib.metadata
import json
from typing import Annotated, Any

import msgspec
import torch
from cozy_runtime.author import (
    AssetBound,
    Config,
    Context,
    FileAsset,
    InvalidRequest,
    Loader,
    Outputs,
    Preflight,
    Telemetry,
    uses_components,
)
from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
from diffusers.modular_pipelines.modular_pipeline import PipelineState
from safetensors.torch import save

from h3 import (
    H3Model,
    H3VideoOutput,
    ReferenceAssets,
    ReferenceMediaToVideoInput,
    preflight_reference_media,
)
from h3 import (
    ref2va as ref2va,
)
from h3_activation_trace import ACTIVE_TRACE, ActivationTrace, FirstStepCaptured
from h3_resident_samples import resident_hashes
from official import (
    NumericalChecks,
    OfficialH3Pipeline,
    ReferencePolicyFacts,
    ScheduleFacts,
    Task,
)


def encode(h3_pipe: Any, pixels: Any, frames: int) -> Any:

    vae = h3_pipe.components["video_vae"]
    pipe = h3_pipe._pipes["t2va"]
    assert all(p.dtype == torch.float32 for p in vae.parameters())
    clip = pixels.permute(2, 0, 1)[None, :, None].expand(-1, -1, frames, -1, -1)
    with torch.no_grad():
        latents = encode_vae_condition(
            vae, clip.contiguous().to(vae.device), pipe.pixel_mean, pipe.pixel_std, encode_seed=42
        )
    assert torch.isfinite(latents).all()
    return latents


def decode(h3_pipe: Any, latents: Any) -> Any:

    state = PipelineState()
    state.set("latents", latents.to(h3_pipe.components["video_vae"].device))
    state.set("output_type", "pt")
    with torch.no_grad():
        return h3_pipe.decode_video("fl2va", state)


def fingerprints(component: str, module: Any) -> dict[str, str]:

    actual, expected = resident_hashes(component, module)
    return {**actual, **{"expected/" + key: value for key, value in expected.items()}}


class VaeModel(H3Model):
    @uses_components("video_vae")
    def encode_probe(self, pixels: Any, frames: int) -> tuple[Any, dict[str, str]]:
        vae = self.pipe.components["video_vae"]
        hashes = fingerprints("video_vae", vae)
        return encode(self.pipe, pixels, frames), hashes

    @uses_components("ref2va_dit")
    def inspect_dit(self) -> dict[str, str]:
        return fingerprints("ref2va_dit", self.pipe.components["ref2va_dit"])

    @uses_components("video_vae")
    def decode_probe(self, latents: Any) -> tuple[Any, dict[str, str]]:
        vae = self.pipe.components["video_vae"]
        return decode(self.pipe, latents), fingerprints("video_vae", vae)

    @uses_components("video_vae")
    def resident_probe(self, pixels: Any, frames: int) -> tuple[Any, dict[str, str]]:
        vae = self.pipe.components["video_vae"]
        before = fingerprints("video_vae", vae)
        video = decode(self.pipe, encode(self.pipe, pixels, frames))
        after = fingerprints("video_vae", vae)
        return video, {
            **{"before/" + k: v for k, v in before.items()},
            **{"after/" + k: v for k, v in after.items()},
        }


def roundtrip(
    model: VaeModel,
    pixels: Any,
    *,
    frames: int,
    cycle: bool,
    ctx: Any,
    tel: Telemetry,
) -> tuple[Any, dict[str, str]]:
    """Observe resident VAE data before and after normal component staging."""
    if frames not in (22, 345):
        raise ValueError("H3 VAE diagnostic supports 22 or 345 frames")
    ctx.raise_if_cancelled()
    if cycle:
        with tel.stage("vae_encode"):
            latents, before = model.encode_probe(pixels, frames)
        for name, value in before.items():
            tel.log("resident tensor fingerprint", phase="before", tensor=name, sha256=value)
        ctx.raise_if_cancelled()
        with tel.stage("resident_dit_hashes"):
            dit = model.inspect_dit()
        for name, value in dit.items():
            tel.log("resident tensor fingerprint", phase="dit", tensor=name, sha256=value)
        ctx.raise_if_cancelled()
        with tel.stage("vae_decode_after_staging"):
            video, after = model.decode_probe(latents)
        hashes = {
            **{"before/" + k: v for k, v in before.items()},
            **{"after/" + k: v for k, v in after.items()},
            **dit,
        }
    else:
        with tel.stage("vae_resident_roundtrip"):
            video, hashes = model.resident_probe(pixels, frames)
    return video, hashes


class TraceResult(msgspec.Struct):
    inference: H3VideoOutput
    activations: Annotated[
        FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))
    ]
    final_latents: Annotated[
        FileAsset, AssetBound(max_bytes=128 << 20, media_types=("application/octet-stream",))
    ]


class ProbeResult(msgspec.Struct):
    activations: Annotated[
        FileAsset, AssetBound(max_bytes=1 << 20, media_types=("application/json",))
    ]
    completed_steps: int


class TracePipeline(OfficialH3Pipeline):
    def denoise(
        self,
        task: Task,
        state: Any,
        *,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks | None = None,
    ) -> ScheduleFacts:

        trace = ACTIVE_TRACE.get()
        if trace is None:
            raise RuntimeError("diagnostic capture context is absent")
        with trace.observe(self.components[f"{task}_dit"]):
            result = super().denoise(
                task, state, on_step=trace.step_callback(on_step), cancel=cancel, checks=checks
            )
        trace.final_latents(state)
        return result


def build_trace_pipeline(config: Config) -> TracePipeline:
    return TracePipeline(config)


class TraceModel(H3Model):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(TracePipeline, factory=build_trace_pipeline)


def trace_preflight(
    payload: ReferenceMediaToVideoInput, assets: ReferenceAssets
) -> ReferencePolicyFacts:
    if payload.seed is None:
        raise InvalidRequest("activation comparisons require an explicit seed", fields=["seed"])
    return preflight_reference_media(payload, assets)


def save_trace(
    trace: Any,
    ctx: Context,
    payload: ReferenceMediaToVideoInput,
    assets: ReferenceAssets,
    model: TraceModel,
    out: Outputs,
) -> FileAsset:

    references = []
    for index in range(len(assets)):
        info = assets.info(index)
        references.append(
            {
                "kind": info.kind,
                "digest": info.digest,
                "size_bytes": info.size_bytes,
                "label": info.label,
                "fidelity": info.fidelity,
            }
        )
    document = trace.document()
    document["provenance"] = {
        "request_id": ctx.request_id,
        "checkpoint": model.checkpoint_ref,
        "inference_package": "minimax-h3",
        "inference_version": importlib.metadata.version("minimax-h3"),
        "diagnostic_module": "h3_diagnostics",
        "torch_version": importlib.metadata.version("torch"),
        "diffusers_version": importlib.metadata.version("diffusers"),
        "runtime_version": importlib.metadata.version("cozy-runtime"),
        "tensorfs_version": importlib.metadata.version("tensorfs"),
        "prompt": payload.prompt,
        "seed": payload.seed,
        "references": references,
        "reference_image_short_edge": payload.reference_image_short_edge,
        "mute": payload.mute,
    }
    raw = json.dumps(document, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > 1 << 20:
        raise ValueError("activation samples exceed the 1 MiB diagnostic limit")
    return out.save_bytes(raw, media_type="application/json")


def reference_trace(
    ctx: Context,
    payload: ReferenceMediaToVideoInput,
    assets: ReferenceAssets,
    facts: Preflight[ReferencePolicyFacts],
    model: TraceModel,
    out: Outputs,
    tel: Telemetry,
) -> TraceResult:
    """Normal H3 video plus bounded activations and original-dtype final latents."""

    trace = ActivationTrace(evaluations=payload.steps, first_step=False)
    with trace.active():
        result = ref2va(ctx, payload, assets, facts, model, out, tel)
    return TraceResult(
        inference=result,
        activations=save_trace(trace, ctx, payload, assets, model, out),
        final_latents=out.save_bytes(save(trace.latents), media_type="application/octet-stream"),
    )


def reference_activations(
    ctx: Context,
    payload: ReferenceMediaToVideoInput,
    assets: ReferenceAssets,
    facts: Preflight[ReferencePolicyFacts],
    model: TraceModel,
    out: Outputs,
    tel: Telemetry,
) -> ProbeResult:
    """Normal preprocessing and first denoise step; no video or final latents."""

    trace = ActivationTrace(evaluations=payload.steps, first_step=True)
    with trace.active():
        try:
            ref2va(ctx, payload, assets, facts, model, out, tel)
        except FirstStepCaptured:
            pass
        else:
            raise RuntimeError("first-step diagnostic did not stop at its callback")
    return ProbeResult(save_trace(trace, ctx, payload, assets, model, out), trace.completed_steps)
