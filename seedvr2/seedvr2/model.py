"""SeedVR2's upstream graph; Runtime owns all learned tensor loading and devices."""

from __future__ import annotations

from typing import Any, cast
from collections.abc import Callable


import torch
from omegaconf import OmegaConf
from torchvision.transforms import functional as vision
from PIL import Image, ImageOps
from cozy_runtime.author import Config, Loader, Model, uses_components

from .upstream.infer import VideoDiffusionInfer
from .upstream.models.dit_v2.nadit import NaDiT
from .upstream.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

DIT = dict(
    vid_in_channels=33,
    vid_out_channels=16,
    vid_dim=2560,
    vid_out_norm="rms",
    txt_in_dim=5120,
    txt_in_norm="layer",
    txt_dim=2560,
    emb_dim=15360,
    heads=20,
    head_dim=128,
    expand_ratio=4,
    norm="rms",
    norm_eps=1e-5,
    ada="single",
    qk_bias=False,
    qk_norm="rms",
    patch_size=(1, 2, 2),
    num_layers=32,
    mm_layers=10,
    mlp_type="swiglu",
    msa_type=None,
    block_type=["mmdit_sr"] * 32,
    window=[(4, 3, 3)] * 32,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
    rope_type="mmrope3d",
    rope_dim=128,
)
VAE = dict(
    act_fn="silu",
    block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownEncoderBlock3D",) * 4,
    up_block_types=("UpDecoderBlock3D",) * 4,
    in_channels=3,
    out_channels=3,
    latent_channels=16,
    layers_per_block=2,
    norm_num_groups=32,
    slicing_sample_min_size=4,
    temporal_scale_num=2,
    inflation_mode="pad",
    spatial_downsample_factor=8,
    temporal_downsample_factor=4,
    use_quant_conv=False,
    use_post_quant_conv=False,
    freeze_encoder=False,
)


class Conditioning(torch.nn.Module):
    positive: torch.Tensor
    negative: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("positive", torch.empty(58, 5120, dtype=torch.bfloat16))
        self.register_buffer("negative", torch.empty(64, 5120, dtype=torch.bfloat16))


def modules() -> dict[str, torch.nn.Module]:
    return {
        "dit": cast(Callable[..., torch.nn.Module], NaDiT)(**DIT).eval().half(),
        "vae": cast(Callable[..., torch.nn.Module], VideoAutoencoderKLWrapper)(**VAE).eval().half(),
    }


class Graph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("architecture") != "SeedVR2-3B":
            raise ValueError("SeedVR2 requires its normalized 3B checkpoint")
        self.components = {**modules(), "conditioning": Conditioning()}


def build(config: Config) -> Graph:
    return Graph(config)


class RestorationModel(Model[Graph]):
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=build)

    @uses_components("dit", "vae", "conditioning")
    def restore(
        self, frames: list[Image.Image], width: int, height: int, seed: int
    ) -> list[Image.Image]:
        device = next(self.graph.components["dit"].parameters()).device
        cfg = OmegaConf.create(
            {
                "vae": {
                    "dtype": "float16",
                    "scaling_factor": 0.9152,
                    "grouping": False,
                    "model": {"temporal_downsample_factor": 4, "spatial_downsample_factor": 8},
                },
                "diffusion": {
                    "schedule": {"type": "lerp", "T": 1000.0},
                    "sampler": {"type": "euler", "prediction_type": "v_lerp"},
                    "timesteps": {
                        "sampling": {"type": "uniform_trailing", "steps": 1},
                        "transform": True,
                    },
                    "cfg": {"scale": 1.0, "rescale": 0.0},
                },
            }
        )
        runner = cast(Any, VideoDiffusionInfer(cfg))
        runner.dit, runner.vae = self.graph.components["dit"], self.graph.components["vae"]
        runner.vae.set_causal_slicing(split_size=4, memory_device="same")
        runner.vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
        runner.configure_diffusion()
        if not frames:
            raise ValueError("SeedVR2 restoration needs at least one frame")
        padded = frames + [frames[-1]] * ((1 - len(frames)) % 4)
        canvas = ((width + 15) // 16 * 16, (height + 15) // 16 * 16)
        left, top = (canvas[0] - width) // 2, (canvas[1] - height) // 2
        pixels = torch.stack(
            [
                vision.to_tensor(
                    ImageOps.expand(
                        frame.resize((width, height), Image.Resampling.LANCZOS),
                        border=(left, top, canvas[0] - width - left, canvas[1] - height - top),
                    )
                )
                for frame in padded
            ]
        )
        pixels = pixels.to(device=device, dtype=torch.float16).permute(1, 0, 2, 3).mul(2).sub(1)
        conditioning = cast(Conditioning, self.graph.components["conditioning"])
        positive, negative = conditioning.positive, conditioning.negative
        with torch.inference_mode(), torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(seed)
            latents = runner.vae_encode([pixels])
            noises = [torch.randn_like(latent) for latent in latents]
            conditions = [
                runner.get_condition(noise, latent_blur=latent, task="sr")
                for noise, latent in zip(noises, latents, strict=True)
            ]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                sample = runner.inference(noises, conditions, [positive], [negative])[0]
            if sample.ndim == 3:
                sample = sample.unsqueeze(1)
            sample = (
                sample[:, : len(frames)]
                .permute(1, 0, 2, 3)
                .float()
                .cpu()
                .clamp(-1, 1)
                .add(1)
                .div(2)
            )
        return [
            vision.to_pil_image(frame).crop((left, top, left + width, top + height))
            for frame in sample
        ]
