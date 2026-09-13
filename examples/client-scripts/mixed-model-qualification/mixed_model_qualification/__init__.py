"""Unpublished, tiny numerical qualification of mixed checkpoint inputs."""

from __future__ import annotations

import hashlib
import math
import struct
import time
from typing import Annotated, Literal

import msgspec
import tensorfs
import torch
from cozy_runtime.author import (
    App,
    Config,
    Context,
    Loader,
    Model,
    ModelArtifact,
    Telemetry,
    WeightsConfig,
    WeightsOutput,
    WeightsPart,
    WeightsSink,
    WeightsTarget,
    WeightsTensor,
    invocable,
    uses_components,
)

BASE_REFERENCE = "paul/cozy-mixed-input-proof@0.0.0-rental-audit.20260913/f32"
app = App()


def write_checkpoint(artifacts: WeightsSink, scale: Literal[2, 3]) -> ModelArtifact:
    """A bounded native checkpoint; no store path, uploaded file or receipt is invented."""
    encoding = dict(tensorfs.seed_digests())["plain/1"]
    tensor = WeightsTensor(
        logical_dtype="f32",
        shape=(2, 2),
        encoding=encoding,
        parts={"value": WeightsPart("f32", (2, 2))},
    )
    with artifacts.open(
        "checkpoint",
        sources={},
        targets={name: WeightsTarget(add={"weight": tensor}) for name in ("alpha", "zeta")},
        configs={"pipeline": WeightsConfig(data=b"{}")},
        order=(("alpha", "weight"), ("zeta", "weight")),
    ) as writer:
        if writer.replayed:
            assert writer.receipt is not None
            return writer.receipt.artifact
        complete = writer.completed_parts
        for name, multiplier in (("alpha", scale), ("zeta", 1)):
            if (name, "weight", "value") not in complete:
                writer.add_part(
                    name, "weight", "value", struct.pack("<4f", multiplier, 0, 0, multiplier)
                )
        writer.add_config("pipeline", b"{}")
        return writer.commit().artifact


@invocable(memoize=True)
async def candidate(ctx: Context, *, artifacts: WeightsSink) -> ModelArtifact:
    ctx.raise_if_cancelled()
    return write_checkpoint(artifacts, 2)


app.job(candidate, weights=(WeightsOutput("checkpoint", max_new_bytes=4096),))


class Matrix(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((2, 2), dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.weight)


class Pipeline:
    def __init__(self) -> None:
        self.components = {"zeta": Matrix(), "alpha": Matrix()}


def pipeline(_config: Config) -> Pipeline:
    return Pipeline()


class TinyModel(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=pipeline)

    @uses_components("alpha", "zeta")
    def measure(self, seed: int, steps: int) -> tuple[float, float, str, list[float]]:
        alpha, zeta = self.pipe.components["alpha"], self.pipe.components["zeta"]
        generator = torch.Generator(device=alpha.weight.device).manual_seed(seed)
        with torch.inference_mode():
            values = torch.rand((1, 2), generator=generator, device=alpha.weight.device) + 1
            input_sum = float(values.sum().item())
            for _ in range(steps):
                values = alpha(zeta(values))
            return (
                input_sum,
                float(values.sum().item()),
                values.device.type,
                torch.randn((2,), generator=generator, device=values.device).tolist(),
            )


class Request(msgspec.Struct, forbid_unknown_fields=True):
    seed: Annotated[int, msgspec.Meta(ge=0, le=2147483647)] = 24680
    steps: Literal[1, 2, 3] = 2
    hold_for_cancel: bool = False


class Result(msgspec.Struct):
    seed: int
    steps: int
    candidate_checkpoint: str
    base_checkpoint: str
    input_sum: float
    candidate_value: float
    base_value: float
    combined_value: float
    device: str
    cpu_rng: str
    cuda_rng: str
    next_noise: list[float]


@app.entrypoint(defaults={"adapter": [{"gpu": "*", "lane": BASE_REFERENCE}]})
def combine(
    ctx: Context,
    payload: Request,
    model: TinyModel,
    adapter: TinyModel,
    tel: Telemetry,
) -> Result:
    ctx.raise_if_cancelled()
    if payload.hold_for_cancel:
        tel.progress(0.5, stage="awaiting-cancel")
        while True:
            ctx.raise_if_cancelled()
            time.sleep(0.1)
    torch.manual_seed(payload.seed)
    input_sum, candidate_value, device, noise = model.measure(payload.seed, payload.steps)
    base_input, base_value, base_device, base_noise = adapter.measure(payload.seed, payload.steps)
    if device != "cuda" or base_device != device:
        raise ValueError("qualification requires actual CUDA execution for both slots")
    if (
        input_sum != base_input
        or noise != base_noise
        or not math.isclose(candidate_value, input_sum * 2**payload.steps, rel_tol=1e-6)
        or not math.isclose(base_value, input_sum * 3**payload.steps, rel_tol=1e-6)
    ):
        raise ValueError(
            "the retained candidate and published base did not execute their distinct weights"
        )
    tel.progress(1.0, stage="mixed-math")
    return Result(
        payload.seed,
        payload.steps,
        model.checkpoint_ref,
        adapter.checkpoint_ref,
        input_sum,
        candidate_value,
        base_value,
        candidate_value + base_value,
        device,
        hashlib.sha256(bytes(torch.random.get_rng_state().tolist())).hexdigest(),
        hashlib.sha256(bytes(torch.cuda.get_rng_state().tolist())).hexdigest(),
        noise,
    )


def result_fields(value: Result) -> dict[str, object]:
    return msgspec.structs.asdict(value)
