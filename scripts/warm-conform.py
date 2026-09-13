#!/usr/bin/env python
"""Warm arms: `Model.warm` runs the dry step the handler used to run under `boot_warmup`.

The runtime calls `warm` once per construction fill, before the placement serves
(model-lifecycle.md #708, se-039). Every arm EXECUTES: the package's own pipeline
constructor over a shrunk copy of its artifact config — every module the real fill holds,
at widths a CPU crosses in seconds — under the real author harness (`for_test` and
`warm_with_fakes`, the executor's own `warm_context`). The component-use scopes the dry
step opens are the ones the runtime would lease; forward hooks on the real modules record
the shapes it ran. No weights, GPU, network, or test framework.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from cozy_runtime.author import Cancelled, Config
from cozy_runtime.author.fakes import warm_with_fakes

ROOT = Path(__file__).resolve().parent.parent

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0


def check(name: str, got: object, expected: object) -> None:
    global _failures
    if got == expected:
        print(f"{PASS}{name}")
        return
    _failures += 1
    print(f"{FAIL}{name}: got {got!r}, expected {expected!r}")


def check_signal(name: str, value: torch.Tensor) -> None:
    check(f"{name} is finite", bool(torch.isfinite(value).all()), True)
    check(f"{name} is nonconstant", bool(value.amax() > value.amin()), True)


def inputs_of(
    module: Any, record: list[tuple[int, ...]], *, output_name: str | None = None,
) -> None:
    """Record every forward's hidden-states shape. A hook records, never invents."""

    def hook(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any], _result: Any) -> None:
        record.append(tuple((args[0] if args else kwargs["hidden_states"]).shape))
        if output_name is not None:
            check_signal(output_name, _result[0])

    module.register_forward_hook(hook, with_kwargs=True)


def decodes_of(
    vae: Any, record: list[tuple[int, ...]], *, output_name: str | None = None,
) -> None:
    """Record every `vae.decode` frame shape — the whole frame, tiled or not."""
    decode = vae.decode

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = decode(*args, **kwargs)
        pixels = result[0] if isinstance(result, tuple) else result.sample
        record.append(tuple(pixels.shape))
        if output_name is not None:
            check_signal(output_name, pixels)
        return result

    vae.decode = spy


def initialize_fixture(pipe: Any) -> None:
    """Fill the CPU lifecycle fixture; production Runtime fills checkpoint weights."""
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for component in pipe.components.values():
            # This arm proves lifecycle and shapes, not the worker's BF16 numerics.
            component.float()
            for name, parameter in component.named_parameters():
                leaf = name.rsplit(".", 1)[-1]
                if leaf == "bias":
                    parameter.zero_()
                elif parameter.ndim == 1 or leaf == "gamma":
                    parameter.fill_(1)
                else:
                    parameter.uniform_(-0.02, 0.02, generator=generator)


def sdxl_config() -> Config:
    """SDXL's exact module tree — HiDiffusion admits the UNet by its keys — at toy widths."""
    clip = {
        "vocab_size": 49408, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 2, "max_position_embeddings": 77,
        "hidden_act": "quick_gelu", "initializer_factor": 1.0, "projection_dim": 32,
    }
    return Config({
        "text_encoder": clip,
        "text_encoder_2": {**clip, "hidden_size": 64, "projection_dim": 64, "hidden_act": "gelu"},
        "unet": {
            "in_channels": 4, "out_channels": 4, "sample_size": 64,
            "block_out_channels": [32, 64, 128], "layers_per_block": 2, "norm_num_groups": 32,
            "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"],
            "up_block_types": ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
            "transformer_layers_per_block": [1, 2, 10], "attention_head_dim": [2, 4, 8],
            "cross_attention_dim": 96, "use_linear_projection": True,
            "addition_embed_type": "text_time", "addition_time_embed_dim": 8,
            "projection_class_embeddings_input_dim": 64 + 6 * 8,
        },
        "vae": {
            "in_channels": 3, "out_channels": 3, "sample_size": 512, "latent_channels": 4,
            "down_block_types": ["DownEncoderBlock2D"] * 4,
            "up_block_types": ["UpDecoderBlock2D"] * 4,
            "block_out_channels": [32, 32, 64, 64], "layers_per_block": 1,
            "norm_num_groups": 32, "scaling_factor": 0.13025, "force_upcast": True,
        },
        "scheduler": {
            "num_train_timesteps": 1000, "beta_start": 0.00085, "beta_end": 0.012,
            "beta_schedule": "scaled_linear", "prediction_type": "epsilon",
            "timestep_spacing": "leading", "steps_offset": 1,
        },
    })


def anima_config() -> Config:
    """The committed Anima configs, narrowed: one layer, 32-wide, the same components."""
    configs = ROOT / "anima" / "configs"
    section = {name: json.loads((configs / f"{name}.json").read_text()) for name in (
        "transformer", "text_encoder", "text_conditioner", "vae", "scheduler",
    )}
    section["transformer"].update(
        num_layers=1, num_attention_heads=2, attention_head_dim=16, adaln_lora_dim=8,
        text_embed_dim=32, encoder_hidden_states_channels=32, crossattn_proj_in_channels=32,
        img_context_dim_out=64,
    )
    section["text_encoder"].update(
        hidden_size=32, head_dim=16, num_attention_heads=2, num_key_value_heads=1,
        intermediate_size=64, num_hidden_layers=1, layer_types=["full_attention"],
        max_window_layers=1,
    )
    section["text_conditioner"].update(
        model_dim=32, source_dim=32, target_dim=32, num_attention_heads=2, num_layers=1
    )
    section["vae"].update(base_dim=16, num_res_blocks=1)
    return Config(section)


def load(name: str) -> Any:
    """The package under proof, from its source tree, in ITS locked environment.

    Loaded when the arm runs, never at module scope: this driver runs inside one package's
    lock, and the other package's stack (sdxl's `hidiffusion`, say) is not in it."""
    sys.path.insert(0, str(ROOT / name))
    return importlib.import_module(name)


def arm_sdxl() -> None:
    package = load("sdxl")
    pipe = package.build_pipeline(sdxl_config())
    unet_inputs: list[tuple[int, ...]] = []
    decoded: list[tuple[int, ...]] = []
    inputs_of(pipe.components["unet"], unet_inputs)
    decodes_of(pipe.components["vae"], decoded)
    model = package.SdxlModel.for_test(pipe=pipe)
    ctx = warm_with_fakes(model)
    check("warm ran without an attempt", ctx.request_id, "")
    check("scopes: encode, one denoise step, decode", [c.method for c in model.harness.calls],
          ["encode", "denoise", "decode"])
    check("every component leased", model.harness.components(),
          ("text_encoder", "text_encoder_2", "unet", "vae"))
    check("one classifier-free UNet step at 512px", unet_inputs, [(2, 4, 64, 64)])
    check("decoded one 512px frame", decoded, [(1, 3, 512, 512)])
    unet = pipe.components["unet"]
    check("HiDiffusion applied for the step", unet._cozy_hidiffusion_active, True)
    arm_cancelled(package.SdxlModel.for_test(pipe=package.build_pipeline(sdxl_config())))


def arm_anima() -> None:
    package = load("anima")
    pipe = package.build_pipeline(anima_config())
    initialize_fixture(pipe)
    dit_inputs: list[tuple[int, ...]] = []
    decoded: list[tuple[int, ...]] = []
    inputs_of(pipe.components["transformer"], dit_inputs, output_name="Anima DiT output")
    decodes_of(pipe.components["vae"], decoded, output_name="Anima decoded frame")
    model = package.AnimaModel.for_test(pipe=pipe)
    ctx = warm_with_fakes(model)
    check("warm ran without an attempt", ctx.request_id, "")
    check("scopes: one render", [c.method for c in model.harness.calls], ["render"])
    check("every component leased", model.harness.components(),
          ("text_encoder", "text_conditioner", "transformer", "vae"))
    check("one DiT step at 512px", dit_inputs, [(1, 16, 1, 64, 64)])
    check("decoded one 512px frame", decoded, [(1, 3, 1, 512, 512)])
    arm_cancelled(package.AnimaModel.for_test(pipe=pipe))


def arm_cancelled(model: Any) -> None:
    """A poisoned construction stops the dry step before it leases anything."""
    try:
        warm_with_fakes(model, cancelled=True)
    except Cancelled:
        check("a cancelled fill refuses before any scope", model.harness.calls, [])
        return
    check("a cancelled fill refuses", "no refusal", "Cancelled")


ARMS = {"sdxl": arm_sdxl, "anima": arm_anima}


def main() -> int:
    selected = sys.argv[1:] or list(ARMS)
    unknown = [name for name in selected if name not in ARMS]
    if unknown:
        print(f"unknown packages: {', '.join(unknown)}", file=sys.stderr)
        return 2
    torch.manual_seed(0)
    for name in selected:
        print(f"{name}:")
        try:
            ARMS[name]()
        except Exception as exc:  # A crashed arm is red, with its exact exception.
            global _failures
            _failures += 1
            print(f"{FAIL}{name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(selected)} packages, {_failures} failures")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
