"""Opt-in plain-CFG batch layout; no retry or device-dependent policy."""

from collections.abc import Callable
from typing import Any

import torch
from diffusers.guiders.classifier_free_guidance import ClassifierFreeGuidance
from diffusers.modular_pipelines.anima.denoise import AnimaLoopDenoiser


def choose_layout(first_block_cache: float) -> tuple[str, str]:
    # Joint-batch residual thresholds do not preserve separate cond/uncond FBC decisions.
    if first_block_cache > 0:
        return "sequential", "preserve separate first-block-cache condition state"
    return "batched", "plain CFG; unchanged guider arithmetic and interval"


class BatchedAnimaLoopDenoiser(AnimaLoopDenoiser):
    """Batch only the exact unmodified plain guider; delegate other guiders unchanged.

    Cosmos repeats its singleton spatial padding mask internally for each batch item.
    Latents/timesteps repeat in guider-state order; predictions split in that same order.
    Batch shape may change kernel selection/rounding. CPU equality is not GPU parity.
    """

    def __init__(self, report: Callable[[str, str], None] | None = None) -> None:
        super().__init__()
        self._report = report
        self._reported: set[tuple[str, str]] = set()

    def _layout(self, layout: str, reason: str) -> None:
        key = layout, reason
        if self._report is not None and key not in self._reported:
            self._report(layout, reason)
            self._reported.add(key)

    @torch.no_grad()
    def __call__(self, components: Any, block_state: Any, i: int, t: Any) -> Any:
        guider = components.guider
        if type(guider) is not ClassifierFreeGuidance:
            self._layout("sequential", "non-plain guider")
            return super().__call__(components, block_state, i, t)
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        states = guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)
        if len(states) != 2:
            self._layout("single", "guider requested one prediction")
            return super().__call__(components, block_state, i, t)
        conditions = {
            key: [getattr(state, key).to(block_state.dtype) for state in states]
            for key in self._guider_input_fields
        }
        # Different context lengths/custom input layouts preserve the existing path;
        # never pad/change attention semantics merely to make a batch fit.
        batch = block_state.latent_model_input.shape[0]
        if (
            block_state.padding_mask.shape[0] != 1
            or block_state.timestep.shape[0] != batch
            or any(
                values[0].shape != values[1].shape or values[0].shape[0] != batch
                for values in conditions.values()
            )
        ):
            self._layout("sequential", "input shapes require original path")
            return super().__call__(components, block_state, i, t)
        self._layout("batched", "two compatible plain-CFG predictions")
        guider.prepare_models(components.transformer)
        try:
            prediction = components.transformer(
                hidden_states=torch.cat([block_state.latent_model_input] * 2, dim=0),
                timestep=torch.cat([block_state.timestep] * 2, dim=0),
                padding_mask=block_state.padding_mask,
                return_dict=False,
                **{key: torch.cat(values, dim=0) for key, values in conditions.items()},
            )[0]
        finally:
            guider.cleanup_models(components.transformer)
        for state, value in zip(states, prediction.split(batch, dim=0), strict=True):
            state.noise_pred = value
        block_state.noise_pred = guider(states)[0]
        return components, block_state
