#!/usr/bin/env python3
"""H3 per-component lane treatments through the real Runtime WeightsSink and TensorFS.

`minimax-h3-tools/scripts/lane-proof.py` proves the DECLARATIONS a lane makes, over the
exact H3 census, on the author surface alone. This driver proves the BYTES and, above all,
OBJECT IDENTITY: it mints a tiny synthetic H3-shaped source in a real `tensorfs.Store`,
derives three lanes from it through the same `WeightsSink` a worker hands the job, and
reads the committed CozyTensors headers back.

The property it exists to establish is the one that decides whether a per-component lane is
affordable at all: a component no lane treats keeps the SOURCE's exact stored objects in
every lane, so N lanes cost the treated bytes N times and the untreated bytes once. In the
served checkpoint that is 58.2 GiB of conditioner and VAE shared across every lane.

No GPU, no rental, no model weights: the fixture is a few hundred kilobytes of random
normals. Run:

    minimax-h3-tools/.venv/bin/python scripts/h3-lane-store-proof.py
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import tensorfs
from cozy_runtime.author import Model, WeightsSink, WeightsTarget
from cozy_runtime.author._model import _derive_model
from cozy_runtime.author.fakes import fake_attempt, fake_context, fake_telemetry
from cozy_runtime.derive.quantization import ArtifactQuantizationRequest
from cozy_runtime.internal.weights_sink import WeightsTransactionHost

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))
from h3_tables import job, lanes  # noqa: E402

SLOT_BYTES = 1 << 24

#: A three-component H3-shaped source. The video VAE row set is the real decoder's shape
#: family — Conv3d (rank 5), norms (rank 1), block-aligned linears (rank 2) and one
#: unaligned projection — because that is what decides which rows a treatment may touch.
LAYOUT: dict[str, dict[str, tuple[int, ...]]] = {
    "text_encoder": {"embed.weight": (512, 64), "norm.weight": (512,)},
    "video_vae": {
        "encoder.conv_in.weight": (4, 4, 2, 2, 2),
        "decoder.transformer_blocks.0.norm1.weight": (128,),
        "decoder.transformer_blocks.0.attn.to_q.weight": (128, 128),
        "decoder.transformer_blocks.0.attn.to_out.0.weight": (128, 128),
        "decoder.proj_in.weight": (128, 24),
    },
    "audio_vae": {
        "decoder.conv_pre.weight_g": (64, 1, 1),
        "decoder.conv_pre.weight_v": (64, 32, 7),
        "pre_block.attn.qkv.weight": (96, 32),
    },
}
RECIPES: dict[str, dict[str, lanes.Treatment]] = {
    "treated": {"video_vae": lanes.Treatment(cast="f16", encode="fp8-rowwise/1")},
    "encoded": {"video_vae": lanes.Treatment(encode="fp8-rowwise/1")},
    "inherited": {},
}


class Source(Model[object]):
    def load(self, loader: Any) -> None:
        del loader


def _fail(what: str) -> None:
    raise SystemExit(f"h3-lane-store-proof: {what}")


def _refuses(what: str, code: str, action: Any) -> None:
    try:
        action()
    except Exception as error:
        detail = getattr(error, "code", "")
        if code and detail != code:
            _fail(f"{what} refused with code {detail!r}, expected {code!r}")
        print(f"  refused {what}: {str(error).splitlines()[0][:110]}")
        return
    _fail(f"{what} did not refuse")


def _plain_spec() -> str:
    return next(digest for alias, digest in tensorfs.seed_digests() if alias == "plain/1")


def _mint(store: Any, layout: dict[str, dict[str, tuple[int, ...]]], suffix: str) -> Any:
    """Write one source checkpoint of real, object-backed float32 payloads."""
    rng = np.random.default_rng(20260908)
    plain = _plain_spec()
    values = {
        component: {
            key: rng.standard_normal(shape).astype(np.float32) for key, shape in rows.items()
        }
        for component, rows in layout.items()
    }
    targets = {
        component: {
            "drop": [],
            "add": {
                key: {
                    "logical_dtype": "f32",
                    "shape": list(array.shape),
                    "encoding": plain,
                    "parts": {"value": {"dtype": "f32", "shape": list(array.shape)}},
                }
                for key, array in rows.items()
            },
        }
        for component, rows in values.items()
    }
    order = [(component, key) for component, rows in values.items() for key in rows]
    writer = store.begin_derived(
        "sha256:" + suffix * 32,
        1,
        {},
        targets,
        {},
        order,
        SLOT_BYTES,
        work_fingerprint="sha256:" + "50" * 32,
    )
    for component, rows in values.items():
        for key, array in rows.items():
            writer.add_part(component, key, "value", io.BytesIO(array.tobytes()))
    manifest = writer.commit()["manifest"]
    return "sha256:" + manifest["sha256"], manifest["length"], values


def _bodies(store: Any, manifest: str) -> dict[tuple[str, str, str], str]:
    """Every (component, tensor, role) -> its stored body, straight off the header."""
    header = tensorfs.parse_header(bytes(store.manifest(manifest)["header"]))
    return {
        (component, key, role): json.dumps(
            part.get("segments", part.get("inline")), sort_keys=True, default=str
        )
        for component, tensors in header["components"].items()
        for key, tensor in tensors.items()
        for role, part in tensor["parts"].items()
    }


def _role_bytes(store: Any, header: Any, component: str, key: str, role: str) -> bytes:
    part = header["components"][component][key]["parts"][role]
    if "inline" in part:
        return bytes(part["inline"])
    return b"".join(
        bytes(store.document("sha256:" + segment["sha256"], segment["length"]))
        for segment in part["segments"]
    )


def _sink(host: Any, model: Any, slots: dict[str, int], attempt: Any) -> WeightsSink:
    """The exact capability a worker hands a job body: public surface over the real host."""
    return WeightsSink(
        attempt,
        {"source": model},
        slots,
        host.open,
        host.structure,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="h3-lane-store-proof-") as root:
        store = tensorfs.Store.init(root)
        manifest, length, values = _mint(store, LAYOUT, "51")
        source_bodies = _bodies(store, manifest)
        model = _derive_model(Source, manifest)
        attempt = fake_attempt("h3-lane-store-proof", spool=Path(root) / "spool")
        ctx = fake_context()
        tel = fake_telemetry(attempt, ctx)
        slots = dict.fromkeys(RECIPES, SLOT_BYTES)
        host = WeightsTransactionHost(
            store=store,
            owner_scope="h3-lane-store-proof",
            request_id="lane-store-proof",
            invocation_spec_digest="sha256:" + "11" * 32,
            work_fingerprint="sha256:" + "22" * 32,
            writer_session_id=1,
            allowed_sources={manifest: length},
            output_bounds=slots | {"overflow": SLOT_BYTES},
        )
        sink = _sink(host, model, slots, attempt)
        structure = sink.structure(model)
        print(f"  synthetic source: {len(source_bodies)} stored roles, manifest {manifest[:23]}…")

        produced: dict[str, str] = {}
        for slot, components in RECIPES.items():
            selections = {
                component: lanes.select(component, treatment, structure.tensors)
                for component, treatment in components.items()
            }
            targets = {
                component: lanes.apply(
                    WeightsTarget(source="source", source_component=component),
                    selections[component],
                )
                if component in selections
                else WeightsTarget(source="source", source_component=component)
                for component in LAYOUT
            }
            with sink.open(
                slot,
                sources={"source": model},
                targets=targets,
                order=tuple((t.component, t.key) for t in structure.tensors),
            ) as transaction:
                for component, selection in selections.items():
                    stats = job._treat(
                        transaction,
                        ctx,
                        tel,
                        ArtifactQuantizationRequest(),
                        selection=selection,
                        source="source",
                    )
                    print(
                        f"  {slot}: {component} {selection.treatment.describe()} -> cast "
                        f"{stats.cast_keys} keys, encoded {stats.encoded_keys} keys, "
                        f"{stats.new_bytes_written} new bytes from {stats.source_bytes_read} read"
                    )
                receipt = transaction.commit()
            facts = json.loads(receipt.tensorfs_receipt)
            produced[slot] = "sha256:" + facts["manifest"]["sha256"]
            observed = facts["inherit_observation"]
            print(
                f"  {slot}: committed, inherited {observed['objects']} objects "
                f"({observed['bytes']} bytes) with {observed['reads']} reads and "
                f"{observed['hashes']} hashes"
            )
            if observed["reads"] or observed["hashes"]:
                _fail(f"{slot} re-read or re-hashed an inherited payload")

        headers = {slot: _bodies(store, identity) for slot, identity in produced.items()}

        # 1. A component no lane names keeps the SOURCE's exact stored objects, in EVERY
        #    lane. This is the property that stops N lanes costing N copies.
        for component in ("text_encoder", "audio_vae"):
            for role, body in source_bodies.items():
                if role[0] != component:
                    continue
                for slot, refs in headers.items():
                    if refs.get(role) != body:
                        _fail(f"{slot} did not inherit the source's exact body for {role}")
        print(
            "  text_encoder and audio_vae bodies are the SAME stored objects in all "
            f"{len(headers)} lanes and in the source"
        )

        # 2. Inside a TREATED component only the treated rows move: the encode-only lane
        #    keeps the Conv3d, the norm and the unaligned projection as the source's own
        #    objects, while a cast rewrites every float32 row, which is what a cast means.
        rows = {role: body for role, body in source_bodies.items() if role[0] == "video_vae"}
        kept = {role for role, body in rows.items() if headers["encoded"].get(role) == body}
        expected = {
            ("video_vae", "encoder.conv_in.weight", "value"),
            ("video_vae", "decoder.transformer_blocks.0.norm1.weight", "value"),
            ("video_vae", "decoder.proj_in.weight", "value"),
        }
        if kept != expected:
            _fail(f"encode-only lane kept {sorted(kept)}, expected {sorted(expected)}")
        if any(headers["treated"].get(role) == body for role, body in rows.items()):
            _fail("the cast lane left a float32 row behind")
        print(
            f"  inside video_vae: encode-only keeps {len(kept)}/{len(rows)} rows as the "
            f"source's own objects (Conv3d, norm, unaligned proj_in); cast+encode moves all "
            f"{len(rows)}"
        )

        # 3. The rows that DID move are correctly declared and carry the right bytes.
        header = tensorfs.parse_header(bytes(store.manifest(produced["treated"])["header"]))
        declared = header["components"]["video_vae"]
        cast = declared["decoder.proj_in.weight"]
        if cast["logical"]["logical_dtype"] != "f16" or set(cast["parts"]) != {"value"}:
            _fail(f"the cast row is {cast['logical']['logical_dtype']} {sorted(cast['parts'])}")
        encoded = declared["decoder.transformer_blocks.0.attn.to_q.weight"]
        if set(encoded["parts"]) != {"data", "scale"} or encoded["encoding"] != job.FP8_SPEC:
            _fail(f"the encoded row is {sorted(encoded['parts'])} under {encoded['encoding']}")
        if encoded["logical"]["logical_dtype"] != "f16":
            _fail(
                "the encoded row's logical dtype is "
                f"{encoded['logical']['logical_dtype']}, expected the lane's cast dtype f16"
            )
        stored = np.frombuffer(
            _role_bytes(store, header, "video_vae", "decoder.proj_in.weight", "value"),
            dtype="<f2",
        )
        want = values["video_vae"]["decoder.proj_in.weight"].reshape(-1).astype(np.float16)
        if not np.array_equal(stored, want):
            _fail("the cast row's stored bytes are not the RNE f16 image of the source")
        print(
            "  decoder.proj_in.weight -> f16 value role whose bytes are the exact RNE image "
            "of the source; attn.to_q.weight -> data+scale under fp8-rowwise/1, logical f16"
        )

        _overflow(store, host, attempt, ctx, tel)
    print("h3-lane-store-proof green")


def _overflow(store: Any, host: Any, attempt: Any, ctx: Any, tel: Any) -> None:
    """An f16 cast that overflows is a typed refusal, never a silently stored inf.

    fp32 reaches 3.4e38 and fp16 stops at 65504, so this is the one cast failure a relative
    round-trip bound cannot see: the error is not large against the tensor's amax, it is
    unrepresentable.
    """
    key = "decoder.transformer_blocks.0.attn.to_q.weight"
    huge = np.full((64, 64), 1e30, dtype=np.float32)
    writer = store.begin_derived(
        "sha256:" + "62" * 32,
        1,
        {},
        {
            "video_vae": {
                "drop": [],
                "add": {
                    key: {
                        "logical_dtype": "f32",
                        "shape": [64, 64],
                        "encoding": _plain_spec(),
                        "parts": {"value": {"dtype": "f32", "shape": [64, 64]}},
                    }
                },
            }
        },
        {},
        [("video_vae", key)],
        SLOT_BYTES,
        work_fingerprint="sha256:" + "63" * 32,
    )
    writer.add_part("video_vae", key, "value", io.BytesIO(huge.tobytes()))
    committed = writer.commit()["manifest"]
    identity, size = "sha256:" + committed["sha256"], committed["length"]
    host.allowed_sources[identity] = size
    model = _derive_model(Source, identity)
    sink = _sink(host, model, {"overflow": SLOT_BYTES}, attempt)
    structure = sink.structure(model)
    selection = lanes.select("video_vae", lanes.Treatment(cast="f16"), structure.tensors)
    target = lanes.apply(WeightsTarget(source="source", source_component="video_vae"), selection)
    with sink.open(
        "overflow",
        sources={"source": model},
        targets={"video_vae": target},
        order=(("video_vae", key),),
    ) as transaction:
        _refuses(
            "an f16 cast of a 1e30 weight",
            "quantization_tripwire",
            lambda: lanes.write_cast(
                transaction,
                ctx,
                tel,
                selection=selection,
                source="source",
                source_component="video_vae",
                target_component="video_vae",
            ),
        )


if __name__ == "__main__":
    main()
