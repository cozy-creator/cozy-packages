"""SeedVR2's upstream graph; Runtime owns all learned tensor loading and devices."""

from __future__ import annotations

from typing import Any, cast
from collections.abc import Callable


import torch
from omegaconf import OmegaConf
from torchvision.transforms import functional as vision
from PIL import Image, ImageOps
from cozy_runtime.author import Config, Loader, Model, Telemetry, uses_components

from .color import Correction, correct
from .frames import extend
from .upstream.infer import VideoDiffusionInfer
from .upstream.models.dit.nadit import NaDiT
from .upstream.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

# SeedVR's configs_7b/main.yaml: separate text/video weights throughout.
DIT = dict(
    vid_in_channels=33,
    vid_out_channels=16,
    vid_dim=3072,
    txt_in_dim=5120,
    txt_dim=3072,
    emb_dim=18432,
    heads=24,
    head_dim=128,
    expand_ratio=4,
    norm="rms",
    norm_eps=1e-5,
    ada="single",
    qk_bias=False,
    qk_norm="rms",
    qk_rope=True,
    patch_size=(1, 2, 2),
    num_layers=36,
    mlp_type="normal",
    shared_mlp=False,
    shared_qkv=False,
    block_type=["mmdit_sr"] * 36,
    window=[(4, 3, 3)] * 36,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 18,
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

ARCHITECTURE = "SeedVR2-7B"


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
        if config.mapping().get("architecture") != ARCHITECTURE:
            raise ValueError("SeedVR2 requires a normalized 7B checkpoint")
        self.components = {**modules(), "conditioning": Conditioning()}


def build(config: Config) -> Graph:
    return Graph(config)


class RestorationModel(Model[Graph], encoded_leaves="accept"):
    """Each phase declares the components it uses, so the Runtime places each on its own:
    the VAE's activations take the room the transformer's weights held while it ran."""

    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=build)

    @uses_components("vae")
    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """One window's pixels, (c, t, h, w) in [-1, 1], to its scaled latent."""
        return _runner(self.graph).vae_encode([pixels])[0]

    @uses_components("dit", "conditioning")
    def denoise(self, latent: torch.Tensor, latent_noise: float) -> torch.Tensor:
        """The one-step restoration of a window's latent, conditioned on itself."""
        runner = _runner(self.graph)
        conditioning = cast(Conditioning, self.graph.components["conditioning"])
        noise = torch.randn_like(latent)
        if latent_noise:
            # Upstream's `_add_noise`, including its (h, w, c) shape for the shift.
            latent = runner.schedule.forward(
                latent,
                noise * 0.1 + torch.randn_like(noise) * 0.05,
                runner.timestep_transform(
                    torch.tensor([1000.0 * latent_noise], device=latent.device),
                    torch.tensor(latent.shape[1:], device=latent.device)[None],
                ),
            )
        condition = runner.get_condition(noise, latent_blur=latent, task="sr")
        with torch.autocast(latent.device.type, dtype=torch.bfloat16):
            return runner.inference(
                [noise], [condition], [conditioning.positive], [conditioning.negative], decode=False
            )[0]

    @uses_components("vae")
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """A restored latent to pixels, (c, t, h, w) in about [-1, 1]."""
        with torch.autocast(latent.device.type, dtype=torch.bfloat16):
            return _runner(self.graph).vae_decode([latent])[0]


def _runner(graph: Graph) -> Any:
    """Upstream's inference driver over the constructed components (configuration only)."""
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
    runner.dit, runner.vae = graph.components["dit"], graph.components["vae"]
    runner.vae.set_causal_slicing(split_size=4, memory_device="same")
    runner.vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
    runner.configure_diffusion()
    return runner


def restore(
    model: RestorationModel,
    frames: list[Image.Image],
    width: int,
    height: int,
    seed: int,
    tel: Telemetry,
    *,
    correction: Correction,
    input_noise: float,
    latent_noise: float,
) -> list[Image.Image]:
    """One window: encode, denoise and decode, each its own component scope."""
    if not frames:
        raise ValueError("SeedVR2 restoration needs at least one frame")
    device = torch.device("cuda", torch.cuda.current_device())
    padded = extend(frames, (1 - len(frames)) % 4)
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
    crop = (slice(top, top + height), slice(left, left + width))
    source = pixels[: len(frames), :, crop[0], crop[1]].mul(2).sub(1)
    pixels = pixels.to(device=device, dtype=torch.float16).permute(1, 0, 2, 3).mul(2).sub(1)
    with torch.inference_mode(), torch.random.fork_rng(devices=[device.index or 0]):
        torch.manual_seed(seed)
        if input_noise:
            pixels = pixels + torch.randn_like(pixels) * (0.025 * input_noise)
        with tel.stage("SeedVR2 VAE encode"):
            latent = model.encode(pixels)
        with tel.stage("SeedVR2 denoise"):
            latent = model.denoise(latent, latent_noise)
        with tel.stage("SeedVR2 VAE decode"):
            sample = model.decode(latent)
        if sample.ndim == 3:
            sample = sample.unsqueeze(1)
        sample = sample[:, : len(frames), crop[0], crop[1]].permute(1, 0, 2, 3).float()
        sample = torch.cat(
            [
                correct(frame[None].clamp(-1, 1), source[index : index + 1].to(device), correction)
                for index, frame in enumerate(sample)
            ]
        )
    return [vision.to_pil_image(frame) for frame in sample.cpu().clamp(-1, 1).add(1).div(2)]
