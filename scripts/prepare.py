#!/usr/bin/env python
"""Build the `quality-judge` artifacts: HF checkpoint -> TensorFS store + artifact config.

Not part of the endpoint and not a test. This is the ARTIFACT WRITER's stand-in: on the
product path a judge checkpoint is an ordinary catalog release that `tfs-003`'s converter
ingested and the hub binds; here the same bytes are written into a local TensorFS store so
the endpoint can be run through `run_slice` on this box with no hub and no coordinator.

It builds the model through the ENDPOINT'S OWN factory (`quality_judge.build_*`) rather
than through a second construction path, so the topology the artifact carries and the
topology the endpoint demands are the same topology by construction — including the rope
tables `persist_buffers` bakes in, which a separate writer would silently omit and the fill
plane would then refuse at serve time.

    nice -n 19 .venv/bin/python scripts/prepare.py            # both families
    nice -n 19 .venv/bin/python scripts/prepare.py judge      # one

Weights live OUTSIDE both repos (`~/cozy_v2/eval-models`) and are never committed.
"""

from __future__ import annotations

import json
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "quality-judge"))

#: tfs-018's ingest extension. The checkpoint WRITER is tfs-003's and is not published yet
#: (the facade wheel exposes reads, `put_file` and snapshots — no header/plan writer), so
#: the artifact is written with the same extension cr-005's SDXL artifact was written with.
#: When tfs-003 publishes a writer this import is the only line that moves.
TFSBENCH = Path("/home/fidika/cozy_v2/tensorfs-bench/lib")

MODELS = Path.home() / "cozy_v2" / "eval-models"
STORE = MODELS / "store"
ARTIFACTS = MODELS / "artifacts"
GIB = 1 << 30
MIB = 1 << 20


@dataclass(frozen=True, slots=True)
class Family:
    """One judge-family artifact: where its bytes come from and what it costs resident."""

    name: str
    source: str
    """The HF snapshot directory under `eval-models`, downloaded at the pinned revision."""
    repo: str
    revision: str
    licence: str
    component: str
    dtype: str
    model_class: str
    vram_bytes: int
    host_bytes: int


FAMILIES: tuple[Family, ...] = (
    Family(
        name="judge",
        source="Qwen3-VL-2B-Instruct",
        repo="Qwen/Qwen3-VL-2B-Instruct",
        revision="89644892e4d85e24eaac8bacfd4f463576704203",
        licence="Apache-2.0",
        component="judge",
        dtype="bfloat16",
        model_class="JudgeModel",
        vram_bytes=6 * GIB,
        host_bytes=2 * GIB,
    ),
    Family(
        name="transcriber",
        source="whisper-small",
        repo="openai/whisper-small",
        revision="973afd24965f72e36ca33b3055d56a652f456b4d",
        licence="MIT",
        component="asr",
        dtype="float16",
        model_class="TranscriberModel",
        vram_bytes=2 * GIB,
        host_bytes=2 * GIB,
    ),
)

#: The vision tower's pixel ceiling, in PIXELS: at patch 16 and merge 2 this is
#: `pixels / 1024` visual tokens per image, so 401 408 px is 392 tokens — three frames of a
#: pairwise comparison then cost ~2 400 tokens of context on an 8 GiB card. The upstream
#: default is 16 777 216 px (16 384 tokens for ONE image), which is a fine default for a
#: 80 GiB server and an OOM here. Recorded in the artifact, not in the endpoint: it is a
#: property of THIS deployment's card, and a bigger card rebinds rather than re-codes.
JUDGE_MAX_PIXELS = 401_408


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


#: `Config`'s carrier rule, restated here because the artifact writer must not WRITE what
#: the endpoint would then be refused for reading. An upstream config carries provenance
#: keys (`_name_or_path`) naming the directory it was exported from, which is exactly the
#: source carrier no factory may see; dropping them is the writer's job, and saying which
#: ones were dropped is how that stays visible instead of silent.
PATH_SUFFIXES = ("_path", "_dir", "_file", "_url", "_repo")


def strip_carriers(value: Any, where: str, dropped: list[str]) -> Any:
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        lowered = str(key).lower()
        is_path = lowered.endswith(PATH_SUFFIXES) or lowered in {"path", "dir", "file", "url"}
        if is_path or (isinstance(item, str) and item.startswith(("/", "./", "../", "~/"))):
            dropped.append(f"{where}.{key}")
            continue
        out[key] = strip_carriers(item, f"{where}.{key}", dropped)
    return out


def blob(path: Path) -> str:
    """A large opaque asset for the artifact config: zlib, then URL-SAFE base64.

    `Config` refuses a string value containing `://` as a source carrier, and a 6 MB
    tokenizer definition contains `://` as an ordinary BPE merge — so a raw embed is
    refused at construction. The encoding also cuts the judge config from 6.2 MB to
    2.6 MB. `quality_judge._blob` is the reader; the two must agree, so they say so.
    """
    import base64
    import zlib

    return base64.urlsafe_b64encode(zlib.compress(path.read_bytes(), 6)).decode("ascii")


def vlm_artifact_config(src: Path) -> dict[str, Any]:
    """The judge artifact's immutable config: model topology + every non-tensor asset.

    A tokenizer, a chat template and an image-preprocessing table are not weights and have
    no checkpoint destinations, so they ride the ONE non-tensor carrier cr-008a's artifact
    has. They are digest-covered exactly like the model config is, which is the property
    that matters: a judge whose chat template moved is a different judge, and the
    construction digest says so.
    """
    tokenizer_config = read_json(src / "tokenizer_config.json")
    chat_template = tokenizer_config.pop("chat_template", "") or read_json(
        src / "chat_template.json"
    )["chat_template"]
    image = read_json(src / "preprocessor_config.json")
    image.pop("processor_class", None)
    image["size"] = {
        "longest_edge": JUDGE_MAX_PIXELS,
        "shortest_edge": image["size"]["shortest_edge"],
    }
    video = read_json(src / "video_preprocessor_config.json")
    video.pop("processor_class", None)
    return {
        "model_config": read_json(src / "config.json"),
        "processor_class": "Qwen3VLProcessor",
        "image_processor": image,
        "video_processor": video,
        "tokenizer_class": tokenizer_config.pop("tokenizer_class", ""),
        "tokenizer_config": _tokenizer_config(tokenizer_config),
        "tokenizer": blob(src / "tokenizer.json"),
        "chat_template": chat_template,
    }


def asr_artifact_config(src: Path) -> dict[str, Any]:
    tokenizer_config = read_json(src / "tokenizer_config.json")
    tokenizer_config.pop("chat_template", None)
    features = read_json(src / "preprocessor_config.json")
    features.pop("processor_class", None)
    return {
        "model_config": read_json(src / "config.json"),
        "generation_config": read_json(src / "generation_config.json"),
        "processor_class": "WhisperProcessor",
        "feature_extractor": features,
        "tokenizer_class": tokenizer_config.pop("tokenizer_class", ""),
        "tokenizer_config": _tokenizer_config(tokenizer_config),
        "tokenizer": blob(src / "tokenizer.json"),
    }


def _tokenizer_config(raw: dict[str, Any]) -> dict[str, Any]:
    """`added_tokens_decoder` is the tokenizer.json's own table restated; passing both
    makes the fast tokenizer re-add every special token it already has."""
    raw.pop("added_tokens_decoder", None)
    raw.pop("tokenizer_file", None)
    return raw


BUILDERS = {"judge": vlm_artifact_config, "transcriber": asr_artifact_config}


# --------------------------------------------------------------------------- safetensors


def st_header(path: Path) -> list[tuple[str, str, list[int], int, int]]:
    """(key, TensorFS dtype, shape, absolute offset, nbytes) in on-disk order."""
    dtypes = {
        "F64": "f64", "F32": "f32", "F16": "f16", "BF16": "bf16",
        "I64": "i64", "I32": "i32", "I16": "i16", "I8": "i8", "U8": "u8", "BOOL": "bool",
    }
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    start = 8 + length
    rows = [
        (key, dtypes[value["dtype"]], list(value["shape"]),
         start + value["data_offsets"][0],
         value["data_offsets"][1] - value["data_offsets"][0])
        for key, value in header.items()
        if key != "__metadata__"
    ]
    rows.sort(key=lambda row: row[3])
    return rows


# --------------------------------------------------------------------------- the build


def materialize(family: Family, config: dict[str, Any], out: Path) -> dict[str, Any]:
    """Build through the endpoint's factory, load the real weights, write safetensors."""
    import torch
    from cozy_runtime.author import Config
    from safetensors.torch import load_file, save_file

    import quality_judge as qj

    factory = qj.build_judge if family.name == "judge" else qj.build_transcriber
    started = time.perf_counter()
    pipeline = factory(Config(config, f"{family.name}.config"))
    module = pipeline.components[family.component]
    built = time.perf_counter() - started

    source_file = MODELS / family.source / "model.safetensors"
    stored = load_file(str(source_file))
    dtype = getattr(torch, family.dtype)
    stored = {k: v.to(dtype) for k, v in stored.items()}
    report = module.load_state_dict(stored, strict=False, assign=True)
    if report.unexpected_keys:
        raise SystemExit(
            f"{family.name}: the checkpoint carries {len(report.unexpected_keys)} keys the "
            f"endpoint's construction has no destination for: {report.unexpected_keys[:6]}"
        )
    # `assign=True` REBINDS every loaded parameter, which severs a weight tie the config
    # declares (`tie_word_embeddings`). Re-tying is not cosmetic here: the artifact is
    # written from `state_dict()`, so an untied `lm_head.weight` would be written as
    # whatever `from_config` randomly initialized. Ties are then materialized as their own
    # destinations at serve time — the fill plane fills destinations and knows nothing
    # about identity — so the tied bytes are stored once (the CAS dedups them) and
    # occupy VRAM twice. Measured on the judge below.
    module.tie_weights()
    state = module.state_dict()
    # Every remaining missing key must be a table `__init__` computed or a key the tie
    # re-bound onto storage the checkpoint DID fill. Anything else is a topology
    # disagreement, and the artifact must not be written over it.
    buffers = {name for name, _ in module.named_buffers()}
    filled = {state[key].untyped_storage().data_ptr() for key in stored if key in state}
    initialized = sorted(set(report.missing_keys) & buffers)
    tied = sorted(
        key
        for key in set(report.missing_keys) - buffers
        if key in state and state[key].untyped_storage().data_ptr() in filled
    )
    stray = set(report.missing_keys) - buffers - set(tied)
    if stray:
        raise SystemExit(f"{family.name}: unfilled non-buffer destinations {sorted(stray)}")
    dense = {key: value.detach().contiguous().clone() for key, value in state.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(dense, str(out))
    total = sum(v.numel() * v.element_size() for v in dense.values())
    return {
        "constructed_seconds": round(built, 2),
        "tensors": len(dense),
        "initialized_buffers": initialized,
        "tied": tied,
        "bytes": total,
        "safetensors_bytes": out.stat().st_size,
    }


def ingest(family: Family, weights: Path) -> dict[str, str]:
    sys.path.insert(0, str(TFSBENCH))
    import tfsbench

    rows = st_header(weights)
    plan = "".join(
        f"{family.component}\t{key}\t{dtype}\t{','.join(map(str, shape))}\t{weights}\t{off}\t{n}\n"
        for key, dtype, shape, off, n in rows
    )
    STORE.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    out = tfsbench.ingest(str(STORE), plan)
    seconds = time.perf_counter() - started
    info = dict(line.split("\t") for line in out.strip().split("\n"))
    info["ingest_seconds"] = f"{seconds:.1f}"
    return info


def build(family: Family) -> dict[str, Any]:
    src = MODELS / family.source
    if not src.is_dir():
        raise SystemExit(f"{src} does not exist — download the pinned revision first")
    target = ARTIFACTS / family.name
    target.mkdir(parents=True, exist_ok=True)
    dropped: list[str] = []
    config = strip_carriers(BUILDERS[family.name](src), "config", dropped)
    if dropped:
        print(f"   dropped source carriers: {dropped}", flush=True)
    (target / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True))
    weights = target / "weights.safetensors"
    print(f"== {family.name}: {family.repo}@{family.revision[:12]} ({family.licence})", flush=True)
    facts = materialize(family, config, weights)
    print(f"   built in {facts['constructed_seconds']}s, {facts['tensors']} tensors, "
          f"{facts['bytes'] / GIB:.3f} GiB", flush=True)
    print(
        f"   initialized buffers {facts['initialized_buffers']}  tied {facts['tied']}",
        flush=True,
    )
    info = ingest(family, weights)
    print(f"   ingested {info['bytes']} B in {info['ingest_seconds']}s "
          f"({info['segments']} segments, {info.get('deduped', '0')} deduped)", flush=True)
    record = {
        "family": family.name,
        "repo": family.repo,
        "revision": family.revision,
        "licence": family.licence,
        "model_class": family.model_class,
        "component": family.component,
        "dtype": family.dtype,
        "store": str(STORE),
        "config": str(target / "config.json"),
        "snapshot": info["checkpoint"],
        "release": f"cozy/quality-judge-{family.name}@ev-003",
        "variant": "sm89",
        "vram_bytes": family.vram_bytes,
        "host_bytes": family.host_bytes,
        "pinned_bytes": 256 * MIB,
        "resident_bytes": facts["bytes"],
        "tensors": facts["tensors"],
        "initialized_buffers": facts["initialized_buffers"],
        "tied": facts["tied"],
    }
    (target / "binding.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(f"   snapshot {info['checkpoint']}", flush=True)
    return record


def main(argv: list[str]) -> int:
    wanted = set(argv[1:]) or {f.name for f in FAMILIES}
    unknown = wanted - {f.name for f in FAMILIES}
    if unknown:
        raise SystemExit(f"unknown families: {sorted(unknown)}")
    free = shutil.disk_usage(MODELS).free
    print(f"store {STORE} — {free / GIB:.0f} GiB free\n", flush=True)
    for family in FAMILIES:
        if family.name in wanted:
            build(family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
