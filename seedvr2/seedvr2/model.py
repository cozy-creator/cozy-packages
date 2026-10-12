"""SeedVR2's upstream graph; Runtime owns all learned tensor loading and devices."""
from __future__ import annotations

from importlib.resources import files

import torch
from omegaconf import OmegaConf
from torchvision.transforms import functional as vision
from PIL import Image
from cozy_runtime.author import Config, Loader, Model, uses_components

from .upstream.infer import VideoDiffusionInfer
from .upstream.models.dit_v2.nadit import NaDiT
from .upstream.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

DIT = dict(
    vid_in_channels=33, vid_out_channels=16, vid_dim=2560, vid_out_norm="rms",
    txt_in_dim=5120, txt_in_norm="layer", txt_dim=2560, emb_dim=15360,
    heads=20, head_dim=128, expand_ratio=4, norm="rms", norm_eps=1e-5,
    ada="single", qk_bias=False, qk_norm="rms", patch_size=(1, 2, 2),
    num_layers=32, mm_layers=10, mlp_type="swiglu", msa_type=None,
    block_type=["mmdit_sr"] * 32, window=[(4, 3, 3)] * 32,
    window_method=["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
    rope_type="mmrope3d", rope_dim=128,
)
VAE = dict(
    act_fn="silu", block_out_channels=(128, 256, 512, 512),
    down_block_types=("DownEncoderBlock3D",) * 4,
    up_block_types=("UpDecoderBlock3D",) * 4,
    in_channels=3, out_channels=3, latent_channels=16, layers_per_block=2,
    norm_num_groups=32, slicing_sample_min_size=4, temporal_scale_num=2,
    inflation_mode="pad", spatial_downsample_factor=8, temporal_downsample_factor=4,
    use_quant_conv=False, use_post_quant_conv=False, freeze_encoder=False,
)


def modules() -> dict[str, torch.nn.Module]:
    return {"dit": NaDiT(**DIT).eval().half(), "vae": VideoAutoencoderKLWrapper(**VAE).eval().half()}


class Graph:
    def __init__(self, config: Config) -> None:
        if config.mapping().get("architecture") != "SeedVR2-3B":
            raise ValueError("SeedVR2 requires its normalized 3B checkpoint")
        self.components = modules()


def build(config: Config) -> Graph:
    return Graph(config)


class RestorationModel(Model[Graph]):
    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=build)

    @uses_components("dit", "vae")
    def restore(self, frames: list[Image.Image], width: int, height: int, seed: int) -> list[Image.Image]:
        device = next(self.graph.components["dit"].parameters()).device
        cfg = OmegaConf.create({
            "vae": {"dtype": "float16", "scaling_factor": 0.9152, "grouping": False,
                    "model": {"temporal_downsample_factor": 4, "spatial_downsample_factor": 8}},
            "diffusion": {"schedule": {"type": "lerp", "T": 1000.0},
                "sampler": {"type": "euler", "prediction_type": "v_lerp"},
                "timesteps": {"sampling": {"type": "uniform_trailing", "steps": 1}, "transform": True},
                "cfg": {"scale": 1.0, "rescale": 0.0}},
        })
        runner = VideoDiffusionInfer(cfg)
        runner.dit, runner.vae = self.graph.components["dit"], self.graph.components["vae"]
        runner.vae.set_causal_slicing(split_size=4, memory_device="same")
        runner.vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
        runner.configure_diffusion()
        padded = frames + [frames[-1]] * ((1 - len(frames)) % 4)
        pixels = torch.stack([vision.to_tensor(frame.resize((width, height), Image.Resampling.LANCZOS)) for frame in padded])
        pixels = pixels.to(device=device, dtype=torch.float16).permute(1, 0, 2, 3).mul(2).sub(1)
        constants = files("seedvr2.upstream")
        # These two fixed upstream prompt encodings are package constants, not fetched models.
        positive = torch.load(constants.joinpath("pos_emb.pt"), map_location=device, weights_only=True)
        negative = torch.load(constants.joinpath("neg_emb.pt"), map_location=device, weights_only=True)
        with torch.inference_mode(), torch.random.fork_rng(devices=[device.index or 0]):
            torch.manual_seed(seed)
            latents = runner.vae_encode([pixels])
            noises = [torch.randn_like(latent) for latent in latents]
            conditions = [runner.get_condition(noise, latent_blur=latent, task="sr") for noise, latent in zip(noises, latents, strict=True)]
            with torch.autocast(device.type, dtype=torch.bfloat16):
                sample = runner.inference(noises, conditions, [positive], [negative])[0]
            sample = sample[:, :len(frames)].permute(1, 0, 2, 3).float().cpu().clamp(-1, 1).add(1).div(2)
        return [vision.to_pil_image(frame) for frame in sample]
