#!/usr/bin/env python
"""CPU proof that `ReferencePlan` thins only a video reference's rows, on the real layout.

Builds a ref2va packed sequence with Diffusers' own layout and timestep-plan builders (an image,
two videos and a continuation's carried rows), tags every row with its packed index, then thins
and releases. Every kept row must keep its exact position, tag and timestep; images, the carried
rows and every generated row must survive; the dropped set must be exactly the clips' frames.
"""

from __future__ import annotations

import sys
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

from minimax_h3.continuation import ContextReferenceLayout  # noqa: E402
from minimax_h3.reference_rows import ReferencePlan, ReferenceRows  # noqa: E402

PATCH = (1, 2, 2)
IMAGE_ROWS, CLIP1, CLIP2 = 12, (7, 8), (12, 2)  # rows; (latent frames, rows per frame)


def packed() -> tuple[list, list, SimpleNamespace]:
    references = [
        MiniMaxH3ImageReference(image=None),
        MiniMaxH3VideoReference(frames=np.zeros((1, 8, 8, 3), np.uint8)),
        MiniMaxH3VideoReference(frames=np.zeros((1, 8, 8, 3), np.uint8)),
        ContextReferenceLayout("video", False),
    ]
    latents = [torch.zeros(1, 24, 1, 8, 6), torch.zeros(1, 24, 7, 4, 8), torch.zeros(1, 24, 12, 2, 4),
               torch.zeros(1, 24, 2, 4, 4)]
    pos, tags, vi, ai, ti, ncv, nca = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.ones(10, dtype=torch.long), references, latents, [], 7, 4, 4, 5, PATCH, 2, 2, 0
    )
    plan = [MiniMaxH3SetTimestepsStep.build_row_timesteps(vi, ai, ncv, nca, ti.numel(), t, a, 0.999, 1.0)
            for t, a in ((0.1, 0.2), (0.5, 0.6))]
    row = torch.arange(tags.numel())
    fields = dict(position_ids=pos, token_tags=tags, video_indices=vi, audio_indices=ai, text_indices=ti,
                  prompt_embeds=row[ti].float()[None, :, None])
    state = SimpleNamespace(**fields, denoiser_input_fields=dict(fields), num_condition_video_rows=ncv,
                            num_condition_audio_rows=nca, latents=row[vi].float()[:, None],
                            audio_latents=row[ai].float()[:, None], row_timestep_plan=plan)
    return references, latents, state


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
    assert torch.equal(ids[state.video_indices[state.num_condition_video_rows :]], generated), "generated"
    assert state.num_condition_video_rows == full.num_condition_video_rows - len(dropped), "condition count"
    assert all(torch.equal(getattr(state, k), v) for k, v in state.denoiser_input_fields.items()), "fields"


def clip_rows(frame_kept: object) -> set[int]:
    vi = packed()[2].video_indices
    first1, first2 = IMAGE_ROWS, IMAGE_ROWS + CLIP1[0] * CLIP1[1]
    return {
        int(vi[first + f * per + r])
        for first, (frames, per) in ((first1, CLIP1), (first2, CLIP2))
        for f in range(frames)
        if not frame_kept(f)
        for r in range(per)
    }


def main() -> None:
    references, latents, state = packed()
    full = packed()[2]
    plan = ReferencePlan.parse("stride=3,steps=1")
    rows = ReferenceRows(references, latents, PATCH, state)
    plan.thin(rows, state)
    check(state, full, clip_rows(lambda f: f % 3 == 0))
    plan.release(rows, state, 0)
    check(state, full, clip_rows(lambda f: f % 3 == 0))
    plan.release(rows, state, 1)
    check(state, full, clip_rows(lambda f: False))
    print("h3 reference rows: thinned and released rows keep their official fields")


if __name__ == "__main__":
    main()
