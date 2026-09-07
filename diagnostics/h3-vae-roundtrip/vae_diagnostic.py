"""Pretrained VAE reconstruction through the serving package and Runtime scopes."""

from typing import Annotated, Any, Literal

import msgspec
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    ImageAsset,
    ImageFrame,
    MediaDecoder,
    Outputs,
    Telemetry,
    VideoAsset,
    uses_components,
)

from h3 import H3Model, _rgb8

app = App()


class Input(msgspec.Struct, forbid_unknown_fields=True):
    image: Annotated[ImageAsset, AssetBound(max_bytes=64 << 20, max_decoded_bytes=256 << 20)]
    frames: Literal[22, 345] = 345


class Result(msgspec.Struct):
    video: VideoAsset
    source: ImageAsset
    reconstruction: ImageAsset
    grid_peak_ratio: float


class VaeModel(H3Model):
    @uses_components("video_vae")
    def reconstruct(self, pixels: Any, frames: int) -> Any:
        import torch
        from diffusers import MiniMaxH3ModularPipeline
        from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

        vae = self.pipe.components["video_vae"]
        assert all(p.dtype == torch.float32 for p in vae.parameters())
        pipe = MiniMaxH3ModularPipeline(blocks=MiniMaxH3VideoDecodeStep())
        pipe.update_components(vae=vae)
        clip = pixels.permute(2, 0, 1)[None, :, None].expand(-1, -1, frames, -1, -1)
        with torch.no_grad():
            latents = encode_vae_condition(
                vae,
                clip.contiguous().to(vae.device),
                pipe.pixel_mean,
                pipe.pixel_std,
                encode_seed=42,
            )
            assert torch.isfinite(latents).all()
            # Execute the same upstream decoder used by the serving package.
            video = pipe(latents=latents, output_type="pt", output="videos")
        return video


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
    with tel.stage("vae_roundtrip"):
        video = model.reconstruct(pixels, payload.frames)
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
    )
