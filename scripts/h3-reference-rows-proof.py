#!/usr/bin/env python
"""CPU proof that only a clip's latent rows leave the real packed layout.

Builds a ref2va sequence with Diffusers' own layout and timestep-plan builders (an image, two
clips and a continuation's carried rows) and tags every row with its packed index. It then
thins the clips to every second latent frame and releases them. Every kept row must keep its
exact position, tag and timestep; the image, the carried rows and every generated row must
survive; each dropped set must be exactly the clips' frames. Then a clip shrinks to `motion_size`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "minimax-h3"))

from minimax_h3.continuation import ContextReferenceLayout
from minimax_h3.reference_rows import FRAME_STRIDE, ClipRows, motion_size

PATCH = (1, 2, 2)
CLIP = MiniMaxH3VideoReference(frames=np.zeros((3, 768, 1344, 3), np.uint8))
REFERENCES = [
    MiniMaxH3ImageReference(image=None),
    CLIP,
    CLIP,
    ContextReferenceLayout("video", False),
]
#: (latent frames, height, width) per image/video entry; the clips' rows follow the image's 12.
LATENTS = [(1, 8, 6), (7, 4, 8), (12, 2, 4), (2, 4, 4)]


def packed() -> SimpleNamespace:
    latents = [torch.zeros(1, 24, *shape) for shape in LATENTS]
    pos, tags, vi, ai, ti, ncv, nca = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(10, dtype=torch.long), REFERENCES, latents, [], 7, 4, 4, 5, PATCH, 2, 2, 0
    )
    plan = [
        MiniMaxH3SetTimestepsStep.build_row_timesteps(
            vi, ai, ncv, nca, ti.numel(), t, a, 0.999, 1.0
        )
        for t, a in ((0.1, 0.2), (0.5, 0.6))
    ]
    row = torch.arange(tags.numel())
    fields = dict(
        position_ids=pos,
        token_tags=tags,
        video_indices=vi,
        audio_indices=ai,
        text_indices=ti,
        prompt_embeds=row[ti].float()[None, :, None],
    )
    return SimpleNamespace(
        **fields,
        denoiser_input_fields=dict(fields),
        num_condition_video_rows=ncv,
        num_condition_audio_rows=nca,
        latents=row[vi].float()[:, None],
        audio_latents=row[ai].float()[:, None],
        row_timestep_plan=plan,
        condition=latents,
    )


def check(state: SimpleNamespace, full: SimpleNamespace, dropped: set[int]) -> None:
    ids = torch.empty(state.token_tags.numel(), dtype=torch.long)
    ids[state.video_indices] = state.latents[:, 0].long()
    ids[state.audio_indices] = state.audio_latents[:, 0].long()
    ids[state.text_indices] = state.prompt_embeds[0, :, 0].long()
    assert set(range(full.token_tags.numel())) - set(ids.tolist()) == dropped, "dropped rows"
    assert torch.equal(state.position_ids, full.position_ids[ids]), "positions"
    assert torch.equal(state.token_tags, full.token_tags[ids]), "tags"
    for (t, i), (ft, fi) in zip(state.row_timestep_plan, full.row_timestep_plan, strict=True):
        assert torch.equal(t[i], ft[fi[ids]]), "timesteps"
    generated = full.video_indices[full.num_condition_video_rows :]
    assert torch.equal(ids[state.video_indices[state.num_condition_video_rows :]], generated)
    assert state.num_condition_video_rows == full.num_condition_video_rows - len(dropped)
    assert all(torch.equal(getattr(state, k), v) for k, v in state.denoiser_input_fields.items())


def clip_rows(dropped_frame: Callable[[int], bool]) -> set[int]:
    vi = packed().video_indices
    spans = ((12, 7, 8), (12 + 7 * 8, 12, 2))  # (first row, latent frames, rows per frame)
    return {
        int(vi[first + f * per + r])
        for first, frames, per in spans
        for f in range(frames)
        if dropped_frame(f)
        for r in range(per)
    }


def main() -> None:
    state, full = packed(), packed()
    clips = ClipRows(state, REFERENCES, state.condition, PATCH)
    clips.thin(state)
    check(state, full, clip_rows(lambda f: f % FRAME_STRIDE != 0))
    clips.release(state)
    check(state, full, clip_rows(lambda f: True))

    assert motion_size(768, 1344) == (384, 672) and motion_size(768, 768) == (384, 384)
    print("h3 reference rows: clip rows leave, every other row keeps its official fields")


if __name__ == "__main__":
    main()
