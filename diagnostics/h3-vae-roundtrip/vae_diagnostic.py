"""Pretrained VAE reconstruction through the serving package and Runtime scopes."""

from typing import Annotated, Any, Literal

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    ImageAsset,
    ImageFrame,
    MediaDecoder,
    Outputs,
    Telemetry,
    VideoAsset,
    uses_components,
)

from h3 import H3Model, _rgb8
from official import TRANSFORMER_EVALUATIONS, NumericalChecks

app = App()


class Input(msgspec.Struct, forbid_unknown_fields=True):
    image: Annotated[ImageAsset, AssetBound(max_bytes=64 << 20, max_decoded_bytes=256 << 20)]
    frames: Literal[22, 345] = 345
    cycle: bool = True


class TensorObservation(msgspec.Struct):
    name: str
    value: str


class Result(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    source: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    reconstruction: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    grid_peak_ratio: float
    hashes: list[TensorObservation]


class ShortInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=4096)]
    seed: int = 24680
    correct_mlp_order: bool = True


class ShortResult(msgspec.Struct):
    video: Annotated[VideoAsset, AssetBound(media_types=("video/mp4",))]
    frame: Annotated[ImageAsset, AssetBound(media_types=("image/png",))]
    tensors: Annotated[FileAsset, AssetBound(media_types=("application/octet-stream",))]
    grid_peak_ratio: float


def encode(h3_pipe: Any, pixels: Any, frames: int) -> Any:
    import torch
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

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
    import torch
    from diffusers.modular_pipelines.modular_pipeline import PipelineState

    state = PipelineState()
    state.set("latents", latents.to(h3_pipe.components["video_vae"].device))
    state.set("output_type", "pt")
    with torch.no_grad():
        return h3_pipe.decode_video("fl2va", state)


def fingerprints(component: str, module: Any) -> dict[str, str]:
    from resident_samples import resident_hashes

    actual, expected = resident_hashes(component, module)
    return {**actual, **{"expected/" + key: value for key, value in expected.items()}}


class VaeModel(H3Model):
    @uses_components("fl2va_dit")
    def sample_short(
        self,
        state: Any,
        *,
        correct_mlp_order: bool,
        on_step: Any,
        cancel: Any,
        checks: NumericalChecks,
    ) -> None:
        import types

        import torch
        from diffusers.models.activations import SwiGLU

        root = self.pipe.components["fl2va_dit"]
        originals = []

        def original_h3_swiglu(activation: Any, value: Any) -> Any:
            gate, linear = activation.proj(value).chunk(2, dim=-1)
            return torch.nn.functional.silu(gate) * linear

        try:
            if correct_mlp_order:
                # Diagnostic only: unchanged original gate/value weight bytes.
                # Production repair belongs to conversion into Diffusers order.
                for activation in root.modules():
                    if not isinstance(activation, SwiGLU):
                        continue
                    originals.append((activation, activation.forward))
                    activation.forward = types.MethodType(original_h3_swiglu, activation)
                assert len(originals) == 52, "expected 50 main and two token-refiner SwiGLUs"
            checks.component("fl2va_dit", root)
            with checks.forwards(root, "fl2va_dit"):
                self.pipe.denoise("fl2va", state, on_step=on_step, cancel=cancel, checks=checks)
        finally:
            for activation, forward in originals:
                activation.forward = forward

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


@app.entrypoint()
def roundtrip(
    ctx: Context,
    payload: Input,
    model: VaeModel,
    decoder: MediaDecoder,
    out: Outputs,
    tel: Telemetry,
) -> Result:
    import numpy as np
    import torch
    from cozy_eval.integrity import output_integrity
    from PIL import Image

    ctx.raise_if_cancelled()
    decoded = decoder.decode_image(payload.image)
    image = Image.frombytes("RGB", (decoded.width, decoded.height), decoded.rgb)
    image.thumbnail((1344, 768), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (1344, 768), (242, 234, 221))
    canvas.paste(image, ((1344 - image.width) // 2, (768 - image.height) // 2))
    pixels = torch.from_numpy(np.asarray(canvas).copy())
    if payload.cycle:
        with tel.stage("vae_encode"):
            latents, before = model.encode_probe(pixels, payload.frames)
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
            video, hashes = model.resident_probe(pixels, payload.frames)
    assert tuple(video.shape) == (1, payload.frames, 3, 768, 1344)
    assert torch.isfinite(video).all()
    ctx.raise_if_cancelled()
    rgb = _rgb8(torch, video)
    facts = output_integrity(rgb.numpy())
    tel.log("VAE diagnostic only", frames=payload.frames, grid_peak_ratio=facts.grid_peak_ratio)
    return Result(
        video=out.save_video(rgb, fps=24),
        source=out.save_image(ImageFrame(1344, 768, canvas.tobytes()), format="png"),
        reconstruction=out.save_image(ImageFrame(1344, 768, bytes(rgb[-1].numpy())), format="png"),
        grid_peak_ratio=float(facts.grid_peak_ratio),
        hashes=[TensorObservation(name, value) for name, value in hashes.items()],
    )


@app.entrypoint()
def short_denoise(
    ctx: Context, payload: ShortInput, model: VaeModel, out: Outputs, tel: Telemetry
) -> ShortResult:
    """124-frame, 384p text-only control with actual conditioning and latent artifacts."""
    import io

    import numpy as np
    import torch
    from cozy_eval.integrity import output_integrity

    ctx.raise_if_cancelled()
    view = model.for_request(ctx, seed=payload.seed)
    state = model.pipe.start_fl2va(
        prompt=payload.prompt,
        first_frame=None,
        last_frame=None,
        generator=model.pipe.generator(view.generator),
    )
    state.set("num_frames", 124)
    state.set("height", 384)
    state.set("width", 672)
    checks = NumericalChecks(tel)
    with tel.stage("condition_text"):
        model.condition_text("fl2va", state, checks=checks)
    prompt_embeds = state.prompt_embeds.detach().float().cpu().numpy().copy()
    with tel.stage("denoise"):
        model.sample_short(
            state,
            correct_mlp_order=payload.correct_mlp_order,
            on_step=tel.step_callback(TRANSFORMER_EVALUATIONS, stage="denoise"),
            cancel=ctx.raise_if_cancelled,
            checks=checks,
        )
    # These are generated conditioning/latent values, not checkpoint weights.
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        prompt_embeds=prompt_embeds,
        video_latents=state.latents.detach().float().cpu().numpy(),
        audio_latents=state.audio_latents.detach().float().cpu().numpy(),
        corrected_mlp_order=np.asarray(payload.correct_mlp_order),
    )
    with tel.stage("decode_video"):
        video = model.decode_video("fl2va", state, checks=checks)
    assert tuple(video.shape) == (1, 124, 3, 384, 672)
    assert torch.isfinite(video).all()
    ctx.raise_if_cancelled()
    pixels = _rgb8(torch, video)
    facts = output_integrity(pixels.numpy())
    tel.log("short denoise diagnostic", grid_peak_ratio=facts.grid_peak_ratio)
    return ShortResult(
        video=out.save_video(pixels, fps=24),
        frame=out.save_image(ImageFrame(672, 384, bytes(pixels[-1].numpy())), format="png"),
        tensors=out.save_bytes(stream.getvalue(), media_type="application/octet-stream"),
        grid_peak_ratio=float(facts.grid_peak_ratio),
    )
