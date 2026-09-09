#!/usr/bin/env python3
"""What each lane DECLARES, proved over the exact H3 census with no weights and no GPU.

The five source components are rebuilt from facts already in this tree: the two DiTs from
the package's own 638-row Diffusers contract plus the one native-only buffer, and the
conditioner and both VAEs from the banked upstream safetensors headers under
`tensorfs/vectors/h3-headers/` — headers only, so not one weight byte is read or fetched.
Every lane's target declaration is then computed by the shipping `job._lane_targets`, and
this proof checks what each lane replaces, what it leaves inherited by reference, that the
declared output ceilings are the ones the shipped descriptor publishes, that a treatment
selects exactly the rows the acceleration lanes ruled on, and every refusal.

The sibling `../../scripts/h3-lane-store-proof.py` proves the same recipes move the right
BYTES and, crucially, that an untreated component keeps the source's exact stored objects
in every lane. That one needs a real store and the real Runtime weights host, so it lives
with the repo's other drivers rather than inside the package.

Run: .venv/bin/python scripts/lane-proof.py
"""

from __future__ import annotations

import glob
import json
import math
import struct
from pathlib import Path
from typing import Any, NoReturn

from cozy_runtime.author import (
    UnsupportedInput,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
    WeightsTarget,
)
from cozy_runtime.derive.quantization import (
    h3_quantization_plan,
    prepare_quantization,
)
from h3_tables import job, lanes
from h3_tables.model_config import parse_production_config
from h3_tables.source import official_full_specs

PROJECT = Path(__file__).resolve().parents[1]


def _headers() -> Path:
    """The banked upstream H3 safetensors headers — metadata only, no weight bytes.

    A worktree lives beside its repo rather than inside the workspace, so the corpus is
    located by walking up from this file.
    """
    for parent in (PROJECT, *PROJECT.parents):
        candidate = parent / "tensorfs/vectors/h3-headers"
        if candidate.is_dir():
            return candidate
    _fail("no tensorfs/vectors/h3-headers corpus found beside this checkout")


LOGICAL = {"F32": "f32", "F16": "f16", "BF16": "bf16"}


def _fail(what: str) -> NoReturn:
    raise SystemExit(f"lane-proof: {what}")


def _refuses(what: str, code: str, action: Any) -> None:
    try:
        action()
    except (UnsupportedInput, ValueError) as error:
        detail = getattr(error, "code", "")
        if code and detail and detail != code:
            _fail(f"{what} refused with code {detail!r}, expected {code!r}")
        print(f"  refused {what}: {str(error).splitlines()[0][:110]}")
        return
    _fail(f"{what} did not refuse")


# ------------------------------------------------------------------ arm A: declarations


def _banked(pattern: str) -> dict[str, dict[str, Any]]:
    corpus = _headers()
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(str(corpus / pattern))):
        raw = Path(path).read_bytes()
        length = struct.unpack("<Q", raw[:8])[0]
        document = json.loads(raw[8 : 8 + length])
        document.pop("__metadata__", None)
        rows.update(document)
    if not rows:
        _fail(f"no banked header rows matched {pattern!r} under {corpus}")
    return rows


def _tensor(component: str, key: str, dtype: str, shape: tuple[int, ...]) -> WeightsSourceTensor:
    return WeightsSourceTensor(
        component=component,
        key=key,
        logical_dtype=dtype,
        shape=shape,
        parts=(WeightsSourcePart("value", dtype, shape),),
        encoding=job.PLAIN_SPEC,
    )


def full_source_structures() -> dict[str, WeightsSource]:
    """The exact FULL H3 source structure, rebuilt from facts already in this tree."""
    sections = parse_production_config(job._asset("model-config.json"))
    dits: list[WeightsSourceTensor] = []
    for component, section in (("fl2va_dit", "transformer"), ("ref2va_dit", "transformer_ref")):
        for key, (dtype, shape) in official_full_specs(sections[section]).items():
            dits.append(_tensor(component, key, dtype, tuple(shape)))
        dits.append(_tensor(component, "rope.inv_freq", "f32", (16,)))
    shared: list[WeightsSourceTensor] = []
    for component, pattern in (
        ("text_encoder", "text_encoder__model-*.safetensors"),
        ("video_vae", "vae__diffusion_pytorch_model-*.safetensors"),
        ("audio_vae", "audio_vae__diffusion_pytorch_model.safetensors"),
    ):
        for key, row in _banked(pattern).items():
            shared.append(
                _tensor(component, key, LOGICAL[row["dtype"]], tuple(row["shape"]))
            )
    census = {
        "fl2va_dit": 639,
        "ref2va_dit": 639,
        "text_encoder": 1058,
        "video_vae": 703,
        "audio_vae": 1087,
    }
    observed = {
        component: sum(1 for t in (*dits, *shared) if t.component == component)
        for component in census
    }
    if observed != census:
        _fail(f"rebuilt source census is {observed}, expected {census}")
    return {
        "dits": WeightsSource(configs=("model",), tensors=tuple(dits)),
        "shared": WeightsSource(configs=("model",), tensors=tuple(shared)),
    }


def _targets(lane: lanes.Lane, granted: dict[str, WeightsSource]) -> dict[str, WeightsTarget]:
    sections = parse_production_config(job._asset("model-config.json"))
    full_targets = job._full_targets()
    plan = prepare_quantization(h3_quantization_plan())
    selections = {
        component: lanes.select(
            component,
            treatment,
            lanes.carried(
                full_targets[component], granted[full_targets[component].source].tensors
            ),
            dit_plan=plan,
        )
        for component, treatment in lane.components.items()
    }
    targets: dict[str, WeightsTarget] = job._lane_targets(
        lane,
        sections,
        job._table_additions(job._topologies(sections), job._plans(job.LAUNCH_SET)),
        full_targets,
        selections,
    )
    return targets


def arm_census() -> None:
    print("arm A — lane declarations over the exact H3 census")
    granted = full_source_structures()
    sections = parse_production_config(job._asset("model-config.json"))
    tables = job._table_additions(job._topologies(sections), job._plans(job.LAUNCH_SET))
    table_rows = {task: len(rows) for task, rows in tables.items()}
    if set(table_rows.values()) != {51}:
        _fail(f"table additions are {table_rows}, expected 51 rows per task")

    inherited_everywhere = {"text_encoder", "video_vae", "audio_vae"}
    for name, lane in lanes.LANES.items():
        targets = _targets(lane, granted)
        if set(targets) != set(lanes.COMPONENTS):
            _fail(f"{name} declares components {sorted(targets)}")
        untouched = {
            component
            for component in inherited_everywhere
            if not targets[component].add and component not in lane.components
        }
        if untouched != inherited_everywhere:
            _fail(f"{name} rewrites a shared component: {sorted(inherited_everywhere - untouched)}")
        dit = targets["fl2va_dit"]
        added = len(dit.add)
        replaced = len(dit.drop)
        print(
            f"  {name:<20} modulation={lane.modulation:<13} "
            f"fl2va_dit +{added} -{replaced}  shared inherited by reference"
        )
        if lane.modulation == "full":
            if added or replaced != 1:
                _fail(f"{name} FULL DiT target is +{added} -{replaced}, expected +0 -1")
        else:
            encoded = 313 if lane.components else 0
            # 51 table rows added; 1 native-only + 106 modulation + the encoded set dropped.
            if added != 51 + encoded or replaced != 1 + 106 + encoded:
                _fail(
                    f"{name} pruned DiT target is +{added} -{replaced}, "
                    f"expected +{51 + encoded} -{1 + 106 + encoded}"
                )

    # The four shipped lanes must be byte-identical declarations to what they always were:
    # the ceilings the descriptor publishes are derived from the catalogue, not typed twice.
    ceilings = {output.name: output.max_new_bytes for output in job.LANE_OUTPUTS}
    shipped = {
        "bf16-full": 65536,
        "bf16-adaln-pruned": 2147614720,
        "fp8-adaln-pruned": 70867091456,
        "mxfp8-adaln-pruned": 70867091456,
    }
    if ceilings != shipped:
        _fail(f"lane ceilings changed: {ceilings}")
    print(f"  lane output ceilings unchanged from the shipped descriptor: {shipped}")

    _component_selection(granted)
    _refusals(granted)


def _component_selection(granted: dict[str, WeightsSource]) -> None:
    print("\n  per-component selection on the real components")
    shared = granted["shared"].tensors

    # The video VAE: h3a-027 §5 predicted 216 decoder ViT linears. The structural selector
    # takes 217 — the 216 block linears PLUS decoder.proj_out.weight [3072, 2048]. It skips
    # decoder.proj_in.weight [2048, 24] on block alignment and every Conv3d on rank.
    video = lanes.select(
        "video_vae",
        lanes.Treatment(cast="f16", encode="fp8-rowwise/1"),
        lanes.carried(job._full_targets()["video_vae"], shared),
    )
    encoded = set(video.encoded)
    if len(encoded) != 217 or any(not key.startswith("decoder.") for key in encoded):
        _fail(f"video_vae selected {len(encoded)} keys, or reached outside the decoder")
    if "decoder.proj_out.weight" not in encoded or "decoder.proj_in.weight" in encoded:
        _fail("video_vae selection is not the exact 216 block linears + proj_out")
    rows = [t for t in shared if t.component == "video_vae"]
    if len(video.cast) != len(rows) - len(encoded):
        _fail(f"video_vae casts {len(video.cast)} of {len(rows)} rows, expected the remainder")
    print(
        f"    video_vae cast=f16 encode=fp8-rowwise/1 -> {len(encoded)} encoded "
        f"(216 block linears + decoder.proj_out.weight), {len(video.cast)} cast, "
        f"{len(rows) - len(video.replaced)} inherited"
    )

    # The conditioner, scoped to the rows the reviewed 902-row target actually carries.
    # h3a-028's ruling: fp8-rowwise/1 over exactly the 350 layers.{0..49} projections, with
    # TEXT_ENCODER_KEEP holding back the token embedding, the 27-block visual tower and the
    # mergers. This arm proves the mechanism expresses that ruling EXACTLY, and it is the
    # one place the two lanes' numbers must agree.
    conditioner = lanes.carried(job._full_targets()["text_encoder"], shared)
    if len(conditioner) != 902:
        _fail(f"the reviewed conditioner target carries {len(conditioner)} rows, expected 902")
    naive = lanes.select("text_encoder", lanes.Treatment(encode="fp8-rowwise/1"), conditioner)
    curated = lanes.select(
        "text_encoder",
        lanes.Treatment(encode="fp8-rowwise/1", keep=lanes.TEXT_ENCODER_KEEP),
        conditioner,
    )
    if "lm_head.weight" in naive.encoded:
        _fail("the selection reached a row the reviewed conditioner target drops")
    if len(naive.encoded) != 441 or len(curated.encoded) != 350:
        _fail(
            f"conditioner selection is {len(naive.encoded)} structural / "
            f"{len(curated.encoded)} kept, expected 441 / 350 (h3a-028)"
        )
    if any(".layers." not in key for key in curated.encoded):
        _fail("the kept conditioner selection reached outside layers.{0..49}")
    sizes = {t.key: math.prod(t.shape) for t in conditioner}
    encoded_bytes = sum(sizes[key] for key in curated.encoded) * 2
    if encoded_bytes != 48_758_784_000:
        _fail(f"conditioner selection is {encoded_bytes} source bytes, expected 48758784000")
    held = sum(sizes[key] for key in lanes.TEXT_ENCODER_KEEP) * 2
    print(
        f"    text_encoder encode=fp8-rowwise/1 -> {len(naive.encoded)} selected structurally, "
        f"{len(curated.encoded)} with TEXT_ENCODER_KEEP ({len(lanes.TEXT_ENCODER_KEEP)} keys); "
        f"{encoded_bytes / 10**9:.2f} GB encoded, {held / 2**30:.2f} GiB of embedding + visual "
        "tower held at bf16 that the shape rule would have taken"
    )
    _refuses(
        "naming linear_fc2 in the keep list (4304 % 32 == 16, never selected)",
        "h3_keep_unmatched",
        lambda: lanes.select(
            "text_encoder",
            lanes.Treatment(
                encode="fp8-rowwise/1",
                keep=(*lanes.TEXT_ENCODER_KEEP, "model.visual.blocks.0.mlp.linear_fc2.weight"),
            ),
            conditioner,
        ),
    )


def _refusals(granted: dict[str, WeightsSource]) -> None:
    print("\n  red arms")
    shared = granted["shared"].tensors
    audio = [t for t in shared if t.component == "audio_vae"]
    representable = [
        t for t in audio if t.key.endswith(".weight") and len(t.shape) == 2 and t.shape[1] % 32 == 0
    ]
    if len(representable) != 6 or any("pre_block" not in t.key for t in representable):
        _fail(f"audio_vae has {len(representable)} structurally selectable rows, expected 6")
    print(
        f"    audio_vae carries {len(representable)} structurally selectable rows "
        f"({', '.join(sorted(t.key for t in representable)[:2])}, ...) — all encoder "
        "pre_block linears, so a shape rule would NOT refuse it"
    )
    _refuses(
        "audio_vae fp8 treatment",
        "h3_component_refused",
        lambda: lanes.select("audio_vae", lanes.Treatment(encode="fp8-rowwise/1"), shared),
    )
    _refuses(
        "audio_vae f16 cast",
        "h3_component_refused",
        lambda: lanes.select("audio_vae", lanes.Treatment(cast="f16"), shared),
    )
    _refuses(
        "a new mxfp8 lane",
        "",
        lambda: lanes.validate_catalogue(
            {"mxfp8-vae-adaln-pruned": lanes.Lane("adaln-pruned", {"video_vae": lanes._DIT_MXFP8})}
        ),
    )
    _refuses(
        "a lane that treats the audio VAE",
        "",
        lambda: lanes.validate_catalogue(
            {"audio-fp8": lanes.Lane("adaln-pruned", {"audio_vae": lanes._DIT_FP8})}
        ),
    )
    _refuses(
        "a FULL lane that treats a component",
        "",
        lambda: lanes.validate_catalogue(
            {"full-cast": lanes.Lane("full", {"video_vae": lanes.Treatment(cast="f16")})}
        ),
    )
    _refuses(
        "a keep entry that excludes nothing",
        "h3_keep_unmatched",
        lambda: lanes.select(
            "video_vae",
            lanes.Treatment(encode="fp8-rowwise/1", keep=("decoder.no_such.weight",)),
            shared,
        ),
    )
    _refuses(
        "a cast that changes nothing",
        "h3_treatment_inert",
        lambda: lanes.select("text_encoder", lanes.Treatment(cast="bf16"), shared),
    )
    _refuses(
        "a treatment that neither casts nor encodes",
        "",
        lambda: lanes.Treatment(),
    )
    _refuses(
        "a lane requesting a repeated output",
        "h3_lanes_repeated",
        lambda: job._requested(job.LaneRequest(lanes=("bf16-full", "bf16-full"))),
    )


def main() -> None:
    arm_census()
    print("\nlane-proof green")


if __name__ == "__main__":
    main()
