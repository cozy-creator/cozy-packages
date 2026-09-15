"""CPU proof of diagnostic hooks on real H3 forwards and two/four-rank exchanges.

No checkpoint, GPU or provider access. This does not establish GPU timing accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from cozy_runtime.author._attention_scope import AttentionLayout, attention_scope
from cozy_runtime.internal.parallel import cp
from diffusers import MiniMaxH3Transformer3DModel
from profile_trace import MARKER, install_profile_hooks
from torch.multiprocessing.spawn import spawn


def rank_main(rank: int, degree: int, directory: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(4)
    dist.init_process_group(
        "gloo", init_method=(directory / "rendezvous").as_uri(), rank=rank, world_size=degree
    )
    try:
        model = MiniMaxH3Transformer3DModel(
            num_attention_heads=4,
            attention_head_dim=128,
            hidden_size=8,
            num_layers=3,
            num_refiner_layers=2,
            ffn_dim=16,
            in_channels=1,
            audio_in_channels=2,
            patch_size=(1, 2, 2),
            text_dim=8,
            freq_dim=8,
            time_embed_hidden_dim=8,
            time_embed_dim=8,
            rope_freq_dim=1,
        ).eval()
        install_profile_hooks(model)
        cp._install_on_component(
            model, degree=degree, comms=cp.CpComms(dist.group.WORLD, rank, torch.device("cpu"))
        )
        packed = 28
        inputs = {
            "hidden_states": torch.randn(1, 12, 4),
            "audio_hidden_states": torch.randn(1, 8, 2),
            "encoder_hidden_states": torch.randn(1, 8, 8),
            "timestep": torch.tensor([1.0]),
            "timestep_indices": torch.zeros(packed, dtype=torch.long),
            "position_ids": torch.zeros(packed, 3, dtype=torch.long),
            "token_tags": torch.tensor([1] * 8 + [2] * 8 + [0] * 12),
            "video_indices": torch.arange(16, 28),
            "text_indices": torch.arange(8),
            "audio_indices": torch.arange(8, 16),
            "return_dict": False,
        }
        layout = AttentionLayout(
            packed, 16, 1, 8, ("token_refiner", "transformer_blocks.0", "transformer_blocks.1")
        )
        with torch.no_grad(), cp.gated_call(), attention_scope(layout):
            expected = model(**inputs)
            actual = model(
                **inputs,
                attention_kwargs={MARKER: {"step": 1, "prefix": str(directory / "profile")}},
            )
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        report = json.loads((directory / f"profile.rank-{rank}.json").read_text())
        assert report["forward_succeeded"] and report["world"] == degree
        exchanges = [row for row in report["spans"] if row["name"] == "qkv_exchange.wait"]
        assert len(exchanges) == 9
        assert all(row["output"]["shape"] == [1, packed, 4 // degree, 128] for row in exchanges)
        assert report["trace_bytes"] > 0 and report["trace_within_bound"]
        with torch.no_grad(), cp.gated_call(), attention_scope(layout):
            after = model(**inputs)
        for left, right in zip(after, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        (directory / f"proof-rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "degree": degree,
                    "equal": True,
                    "qkv_exchanges": len(exchanges),
                    "trace_bytes": report["trace_bytes"],
                }
            )
            + "\n"
        )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=(2, 4), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run: Any = spawn
    run(rank_main, args=(args.degree, args.out), nprocs=args.degree, join=True)
    print(
        json.dumps(
            {
                "degree": args.degree,
                "ranks": [
                    json.loads((args.out / f"proof-rank-{rank}.json").read_text())
                    for rank in range(args.degree)
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
