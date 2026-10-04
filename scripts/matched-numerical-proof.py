"""CPU equation/precision proof; synthetic predictions are not inference qualification."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import diffusers
import torch
from diffusers import (
    AnimaAutoBlocks,
    AnimaModularPipeline,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.models.embeddings import Timesteps
from diffusers.modular_pipelines.anima.denoise import AnimaLoopBeforeDenoiser
from diffusers.modular_pipelines import ModularPipelineBlocks


def symbols(source, names, scope):
    tree = ast.parse(source)
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<reviewed-source>", "exec"), scope)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--sdxl-config", type=Path, required=True)
    p.add_argument("--anima-config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    sources = {
        name: (root / name).read_text()
        for name in ("sdxl/sdxl/__init__.py", "anima/anima/__init__.py")
    }
    for name in (
        "comfy/model_sampling.py",
        "comfy/samplers.py",
        "comfy/sample.py",
        "comfy/ldm/modules/diffusionmodules/util.py",
    ):
        sources[name] = subprocess.check_output(
            ["git", "-C", str(args.comfy), "show", f"{args.commit}:{name}"], text=True
        )
    scope = {
        "torch": torch,
        "math": math,
        "Any": Any,
        "ClassifierFreeGuidance": ClassifierFreeGuidance,
        "AnimaLoopBeforeDenoiser": AnimaLoopBeforeDenoiser,
    }
    symbols(sources["sdxl/sdxl/__init__.py"], ["_initial_noise_scale"], scope)
    symbols(sources["anima/anima/__init__.py"], ["_Fp32Guidance", "_Fp32Timestep"], scope)
    symbols(
        sources["comfy/model_sampling.py"],
        [
            "reshape_sigma",
            "EPS",
            "CONST",
            "time_snr_shift",
            "ModelSamplingDiscrete",
            "ModelSamplingDiscreteFlow",
        ],
        scope,
    )
    symbols(
        sources["comfy/samplers.py"],
        ["normal_scheduler", "ddim_scheduler", "Sampler", "cfg_function"],
        scope,
    )
    symbols(sources["comfy/sample.py"], ["prepare_noise_inner", "prepare_noise"], scope)
    symbols(sources["comfy/ldm/modules/diffusionmodules/util.py"], ["make_beta_schedule"], scope)
    results = []
    # Exercise the actual production denoise input conversion and maintained
    # block assembly, without constructing model weights or a CUDA context.
    owner = next(
        node
        for node in ast.parse(sources["sdxl/sdxl/__init__.py"]).body
        if getattr(node, "name", None) == "SdxlModel"
    )
    method = next(node for node in owner.body if getattr(node, "name", None) == "denoise")
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<production-denoise>", "exec"), scope)

    class InputProbe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float16))

        def begin_denoise_request(self, *args):
            pass

        def forward(self, latents, *args, **kwargs):
            assert latents.dtype == torch.float16
            return SimpleNamespace(sample=latents)

    probe = SimpleNamespace(pipe=SimpleNamespace(components={"unet": InputProbe()}))
    result = scope["denoise"](
        probe,
        torch.ones((2, 4, 2, 2), dtype=torch.float32),
        torch.tensor(999.0),
        None,
        None,
        None,
        0,
        20,
        False,
    )
    assert result.dtype == torch.float16
    results.append(
        {"check": "production_sdxl_denoiser_input", "input": "float32", "compute": "float16"}
    )
    scope.update(
        {
            "AnimaAutoBlocks": AnimaAutoBlocks,
            "AnimaModularPipeline": AnimaModularPipeline,
            "ModularPipelineBlocks": ModularPipelineBlocks,
            "_Phases": Any,
            "_progress_bar": lambda phases: None,
        }
    )
    symbols(sources["anima/anima/__init__.py"], ["_text2image_pipeline"], scope)
    pipeline = scope["_text2image_pipeline"](torch.device("cpu"), SimpleNamespace())
    before = pipeline.blocks.sub_blocks["denoise.denoise"].sub_blocks["before_denoiser"]
    assert isinstance(before, scope["_Fp32Timestep"])
    pipeline.register_components(guider=scope["_Fp32Guidance"]())
    assert isinstance(pipeline.guider, ClassifierFreeGuidance)
    results.append({"check": "production_anima_block_assembly", "fp32_timestep_and_guidance": True})
    configs = {
        name: json.loads(path.read_text())
        for name, path in (("sdxl", args.sdxl_config), ("anima", args.anima_config))
    }
    for seed in (0, 1114002, -1, (1 << 63) - 1):
        for shape in ((1, 4, 128, 128), (1, 16, 128, 128)):
            with torch.random.fork_rng(devices=[]):
                reference = scope["prepare_noise"](torch.empty(shape), seed)
            product = torch.randn(
                shape,
                dtype=torch.float32,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            )
            assert torch.equal(reference, product)
            results.append({"check": "cpu_noise", "seed": seed, "shape": shape, "exact": True})
    for model in ("sdxl", "anima"):
        config = configs[model]
        if model == "sdxl":
            cls, steps, dtype, prediction = (
                EulerDiscreteScheduler,
                20,
                torch.float16,
                scope["EPS"](),
            )
            sampling = scope["ModelSamplingDiscrete"](
                SimpleNamespace(
                    sampling_settings={
                        "beta_schedule": "linear",
                        "linear_start": config["beta_start"],
                        "linear_end": config["beta_end"],
                        "timesteps": config["num_train_timesteps"],
                    }
                )
            )
            comfy_sigmas = scope["ddim_scheduler"](sampling, steps)
        else:
            cls, steps, dtype, prediction = (
                FlowMatchEulerDiscreteScheduler,
                30,
                torch.bfloat16,
                scope["CONST"](),
            )
            sampling = scope["ModelSamplingDiscreteFlow"](
                SimpleNamespace(sampling_settings={"shift": config["shift"], "multiplier": 1.0})
            )
            comfy_sigmas = scope["normal_scheduler"](sampling, steps)
        scheduler = cls.from_config(config)
        model_sigma_max = float(scheduler.sigmas.max())
        scheduler.set_timesteps(steps)
        assert float((scheduler.sigmas - comfy_sigmas).abs().max()) <= 1e-5
        if model == "sdxl":
            for sigmas in (
                scheduler.sigmas,
                torch.tensor([model_sigma_max, 0]),
                torch.tensor([model_sigma_max * 1.1, 0]),
            ):
                wrap = SimpleNamespace(
                    inner_model=SimpleNamespace(
                        model_sampling=SimpleNamespace(sigma_max=model_sigma_max)
                    )
                )
                max_denoise = scope["Sampler"]().max_denoise(wrap, sigmas)
                original = prediction.noise_scaling(
                    sigmas[0], torch.ones((1, 4, 1, 1)), torch.zeros((1, 4, 1, 1)), max_denoise
                )
                actual = scope["_initial_noise_scale"](sigmas, model_sigma_max)
                assert torch.equal(original, actual.expand_as(original))
            results.append(
                {"check": "eps_initial_scaling", "starts_below_at_and_above_model_maximum": True}
            )
        else:
            fixed = scope["_Fp32Timestep"]()
            original = AnimaLoopBeforeDenoiser()
            differences = []
            for t, sigma in zip(scheduler.timesteps, comfy_sigmas[:-1], strict=True):
                components = SimpleNamespace(scheduler=scheduler)
                state = SimpleNamespace(latents=torch.ones((1, 16, 1, 2, 2)), dtype=torch.bfloat16)
                fixed(components, state, 0, t)
                assert (
                    state.timestep.dtype == torch.float32
                    and state.latent_model_input.dtype == torch.bfloat16
                )
                expected = sampling.timestep(sigma).reshape(1)
                assert float((state.timestep - expected).abs().max()) <= 2e-7
                embedding = Timesteps(2048, flip_sin_to_cos=True, downscale_freq_shift=0)
                assert float((embedding(state.timestep) - embedding(expected)).abs().max()) <= 2e-7
                original(components, state, 0, t)
                differences.append(float((state.timestep.float() - expected).abs().max()))
            results.append(
                {
                    "check": "anima_fp32_timestep",
                    "steps": steps,
                    "legacy_max_timestep_difference": max(differences),
                }
            )
        # Isolate equation and precision from tiny grid-generation differences.
        # Both paths use this scheduler's actual grid and identical low-precision
        # synthetic network predictions. This does not compare real networks.
        for guidance in (1.0, 4.5, 7.0):
            for magnitude in (1.0, 1000.0):
                scheduler = cls.from_config(config)
                scheduler.set_timesteps(steps)
                generator = torch.Generator().manual_seed(37)
                initial = torch.randn((1, 4, 4, 4), generator=generator) * magnitude
                actual = initial.clone()
                expected = initial.clone()
                legacy = initial.to(dtype)
                legacy_scheduler = cls.from_config(config)
                legacy_scheduler.set_timesteps(steps)
                guider = scope["_Fp32Guidance"](guidance_scale=guidance)
                differences = []
                for index, t in enumerate(scheduler.timesteps):
                    cond = (torch.randn(initial.shape, generator=generator) * magnitude).to(dtype)
                    uncond = (torch.randn(initial.shape, generator=generator) * magnitude).to(dtype)
                    sigma = scheduler.sigmas[index]
                    if guidance == 1:
                        guided = cond.float()
                        guided_legacy = cond
                    elif model == "anima":
                        guider.set_state(step=index, num_inference_steps=steps, timestep=t)
                        guided = guider.forward(cond, uncond).pred
                        guided_legacy = uncond + guidance * (cond - uncond)
                    else:
                        guided = uncond.float() + guidance * (cond.float() - uncond.float())
                        guided_legacy = uncond + guidance * (cond - uncond)
                    if model == "sdxl":
                        scheduler.scale_model_input(actual, t)
                        legacy_scheduler.scale_model_input(legacy, t)
                    actual = scheduler.step(guided, t, actual).prev_sample
                    legacy = legacy_scheduler.step(guided_legacy, t, legacy).prev_sample
                    cond_x0 = prediction.calculate_denoised(sigma, cond.float(), expected)
                    uncond_x0 = prediction.calculate_denoised(sigma, uncond.float(), expected)
                    # Empty post-CFG hooks: execute the actual stock CFG function.
                    denoised = (
                        cond_x0
                        if guidance == 1
                        else scope["cfg_function"](
                            None, cond_x0, uncond_x0, guidance, expected, sigma
                        )
                    )
                    expected = expected + ((expected - denoised) / sigma) * (
                        scheduler.sigmas[index + 1] - sigma
                    )
                    assert actual.dtype == torch.float32
                    differences.append(float((actual - expected).abs().max()))
                scale = max(1.0, float(expected.abs().max()))
                relative = max(differences) / scale
                assert relative <= 2e-6
                legacy_difference = float((legacy.float() - actual).abs().max())
                results.append(
                    {
                        "check": "full_scheduler_fp32",
                        "model": model,
                        "steps": steps,
                        "guidance": guidance,
                        "prediction_magnitude": magnitude,
                        "max_absolute_difference_from_stock_euler": max(differences),
                        "scaled_error": relative,
                        "legacy_state_dtype": str(legacy.dtype),
                        "legacy_state_finite": bool(torch.isfinite(legacy).all()),
                        "legacy_final_difference": legacy_difference
                        if math.isfinite(legacy_difference)
                        else None,
                    }
                )
    result = {
        "scope": "source CPU synthetic equation and precision proof; not inference, GPU, quality or timing",
        "comfy_commit": args.commit,
        "diffusers": diffusers.__version__,
        "torch": torch.__version__,
        "source_sha256": {
            name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()
        },
        "configs": configs,
        "checks": results,
    }
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": len(results), "scope": result["scope"]}))


if __name__ == "__main__":
    main()
