#!/usr/bin/env python
"""SDXL's HiDiffusion UNet on the CPU, tiny but keyed as SDXL: a UNet call at one step, and one
RAU-Net block within it, gives the same output and the same switch however often it runs and
whatever the RNG did before it (the Runtime's out-of-memory recovery runs a call or a block
again), and the switch follows the step.

Run with the package's own interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sdxl"))

from sdxl import _hidiffusion_unet_type  # noqa: E402

#: SDXL's module keys (HiDiffusion recognizes the UNet by them) at a few channels.
CONFIG = {
    "sample_size": 32,
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    "block_out_channels": (32, 64, 96),
    "transformer_layers_per_block": (1, 2, 10),
    "attention_head_dim": (2, 4, 4),
    "cross_attention_dim": 32,
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 8,
    "projection_class_embeddings_input_dim": 32 + 6 * 8,
}
STEPS = 10


def main() -> None:
    torch.manual_seed(0)
    unet: Any = _hidiffusion_unet_type().from_config(CONFIG).eval()
    unet.name_or_path, unet._cozy_hidiffusion_active = "", False
    unet._cozy_base_num_upsamplers = unet.num_upsamplers
    unet.begin_denoise_request(STEPS, True, 7)
    x, text = torch.randn(2, 4, 64, 64), torch.randn(2, 77, 32)
    conditions = {"text_embeds": torch.randn(2, 32), "time_ids": torch.tensor([[512.0] * 6] * 2)}

    def call(step: int, history: int) -> tuple[torch.Tensor, Any]:
        unet.cozy_step, unet.info["upsample_size"] = step, None
        torch.manual_seed(history)  # whatever ran before it
        out = unet(x, 999 - 99 * step, encoder_hidden_states=text, added_cond_kwargs=conditions)
        return out.sample, unet.info["upsample_size"]

    # One RAU-Net block, run again inside the step with the inputs the UNet gave it.
    block = next(m for m in unet.modules() if type(m).__name__ == "cross_attn_down_block")
    seen: dict[str, Any] = {}
    block.register_forward_pre_hook(lambda _, a, k: seen.update(a=a, k=k), with_kwargs=True)

    def rerun() -> tuple[torch.Tensor, Any]:
        unet.info["upsample_size"] = None
        return block(*seen["a"], **seen["k"])[0], unet.info["upsample_size"]

    switches = []
    with torch.inference_mode():
        for step in range(STEPS):
            first, switch = call(step, step)
            again, switch_again = call(step, 1000 + step)
            assert torch.equal(first, again), f"step {step}: a call run again differs"
            assert switch == switch_again, f"step {step}: the RAU-Net switch moved"
            once, block_switch = rerun()
            twice, block_switch_again = rerun()
            assert torch.equal(once, twice), f"step {step}: a block run again differs"
            assert block_switch == block_switch_again == switch, f"step {step}: its switch moved"
            switches.append(switch)
    assert any(switches) and not all(switches), f"the switch never moved: {switches}"
    print("sdxl model proof: a HiDiffusion call at a step runs the same however often it runs")


if __name__ == "__main__":
    main()
