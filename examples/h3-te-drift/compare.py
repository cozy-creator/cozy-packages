"""Compare `encode_text` outputs of several lanes against one reference lane, on this computer.

    python compare.py prompt-names.json bf16=bf16/<run>-states.bin fp8=... mxfp8=...

The first lane is the reference. Prints per hidden state, across prompts: mean and worst
per-token cosine, mean and worst relative L2, the mean relative L2 without token 0 (Qwen's
massive-activation token) and the max abs error; then the final state per prompt.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.double(), b.double()
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    diff = a - b
    return {
        "cos_mean": cos.mean().item(),
        "cos_min": cos.min().item(),
        "rel_l2": (diff.norm() / a.norm()).item(),
        "rel_l2_no_tok0": (diff[1:].norm() / a[1:].norm()).item(),
        "max_abs": diff.abs().max().item(),
    }


def main() -> None:
    names = json.loads(Path(sys.argv[1]).read_text())
    lanes = dict(arg.split("=", 1) for arg in sys.argv[2:])
    files = {lane: load_file(path) for lane, path in lanes.items()}
    reference_lane, reference = next(iter(files.items()))
    layers = sorted({int(k.rsplit(".", 1)[1]) for k in reference if ".hidden_states." in k})
    report = {}
    for lane, states in list(files.items())[1:]:
        for i, name in enumerate(names):
            if not torch.equal(states[f"{i}.token_ids"], reference[f"{i}.token_ids"]):
                raise SystemExit(f"{lane} tokenized prompt {name} differently")
        rows = {
            name: {
                k: compare(reference[f"{i}.hidden_states.{k}"], states[f"{i}.hidden_states.{k}"])
                for k in layers
            }
            for i, name in enumerate(names)
        }
        summary = {}
        for k in layers:
            cells = [rows[name][k] for name in names]
            summary[k] = {
                "cos_mean": sum(c["cos_mean"] for c in cells) / len(cells),
                "cos_min": min(c["cos_min"] for c in cells),
                "rel_l2_mean": sum(c["rel_l2"] for c in cells) / len(cells),
                "rel_l2_max": max(c["rel_l2"] for c in cells),
                "rel_l2_no_tok0_mean": sum(c["rel_l2_no_tok0"] for c in cells) / len(cells),
                "max_abs": max(c["max_abs"] for c in cells),
            }
        report[lane] = {
            "summary": summary,
            "final_per_prompt": {n: rows[n][layers[-1]] for n in names},
        }
        print(f"== {lane} vs {reference_lane}")
        for k, s in summary.items():
            print(
                f"  hidden_states[{k:>2}] cos mean {s['cos_mean']:.5f} min {s['cos_min']:.4f}  "
                f"rel L2 mean {s['rel_l2_mean']:.4f} max {s['rel_l2_max']:.4f}  "
                f"w/o token 0 {s['rel_l2_no_tok0_mean']:.4f}  max abs {s['max_abs']:.1f}"
            )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
