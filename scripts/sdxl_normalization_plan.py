"""Reproduce the SDXL normalization data from the pinned upstream key/layout mappings."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
from cozy_runtime.author import Config
from cozy_runtime.author._loader import census
from diffusers.loaders.single_file_utils import (
    convert_ldm_clip_checkpoint,
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_open_clip_checkpoint,
)

from sdxl import SdxlPipeline
from sdxl.normalization import NormalizationPlan, TensorRoute

LiteralKind = Literal["graft", "read", "transpose"]

PREFIXES = {
    "unet": "model.diffusion_model.",
    "vae": "first_stage_model.",
    "text_encoder": "conditioner.embedders.0.transformer.",
    "text_encoder_2": "conditioner.embedders.1.model.",
}


def strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))


@dataclass(frozen=True)
class Route:
    """A source coordinate transform; no tensor values or fake model output exist here.

    The upstream mapping helpers move/slice/transpose dictionary values. Copy/detach do
    not change those coordinates; actual bytes are read only by the native job.
    """

    component: str
    key: str
    shape: tuple[int, ...]
    offset: int
    strides: tuple[int, ...]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def T(self) -> Route:
        assert self.ndim == 2
        return replace(self, shape=self.shape[::-1], strides=self.strides[::-1])

    def contiguous(self) -> Route:
        return self

    def clone(self) -> Route:
        return self

    def detach(self) -> Route:
        return self

    def __getitem__(self, key: int | slice | tuple[int | slice, ...]) -> Route:
        index = key if isinstance(key, tuple) else (key,)
        index = (*index, *([slice(None)] * (self.ndim - len(index))))
        shape, stride, offset = [], [], self.offset
        for size, step, item in zip(self.shape, self.strides, index, strict=True):
            if isinstance(item, int):
                assert 0 <= item < size
                offset += item * step
            else:
                start, stop, jump = item.indices(size)
                assert jump > 0
                shape.append(len(range(start, stop, jump)))
                stride.append(step * jump)
                offset += start * step
        return replace(self, shape=tuple(shape), strides=tuple(stride), offset=offset)


def routes_for(plan: NormalizationPlan) -> tuple[TensorRoute, ...]:
    checkpoint = {
        PREFIXES[component] + key: Route(component, key, spec.shape, 0, strides(spec.shape))
        for component, rows in plan.source.items()
        for key, spec in rows.items()
    }
    # The actual family constructor runs on meta: no source weight or GPU allocation.
    with torch.device("meta"):
        model = SdxlPipeline(Config(plan.configs))
    expected = census(model)
    mapped: dict[str, dict[str, Route]] = {
        "unet": convert_ldm_unet_checkpoint(checkpoint, plan.configs["unet"]),
        "vae": convert_ldm_vae_checkpoint(checkpoint, plan.configs["vae"]),
        "text_encoder": convert_ldm_clip_checkpoint(checkpoint),
        "text_encoder_2": convert_open_clip_checkpoint(
            model.components["text_encoder_2"], checkpoint, prefix=PREFIXES["text_encoder_2"]
        ),
    }
    # Transformers5's CLIPTextModel exposes the text tower directly; the projection
    # model still contains text_model. This mapping targets that one pinned constructor.
    mapped["text_encoder"] = {
        key.removeprefix("text_model."): value for key, value in mapped["text_encoder"].items()
    }
    rows = []
    used = set()
    for destination in expected.destinations:
        component = destination.component
        key = destination.key.removeprefix(component + ".")
        route = mapped[component].pop(key)
        assert route is not None and route.shape == destination.spec.shape, (component, key)
        spec = plan.source[route.component][route.key]
        assert spec.dtype == destination.spec.dtype and route.component == component
        kind: LiteralKind = "graft"
        if route.strides == strides(route.shape):
            if route.shape != spec.shape or route.offset:
                kind = "read"
        else:
            assert len(spec.shape) == 2 and route.offset == 0
            assert route.strides == (1, spec.shape[1])
            kind = "transpose"
        rows.append(TensorRoute(component, key, route.key, route.shape, kind, route.offset))
        used.add((component, route.key))
    assert [(c, k) for c, values in mapped.items() for k in values] == [
        ("text_encoder", "embeddings.position_ids")
    ]
    unused = {(c, k) for c, values in plan.source.items() for k in values if (c, k) not in used}
    assert unused == {
        ("text_encoder", "text_model.embeddings.position_ids"),
        ("text_encoder_2", "logit_scale"),
    }
    return tuple(rows)
