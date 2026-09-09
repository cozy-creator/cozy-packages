#!/usr/bin/env python3
"""Prove table metadata upgrades and no-cast restamp behavior without GPU/weight I/O."""

from __future__ import annotations

import copy
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from typing import Any

from cozy_runtime.author import (
    ConformanceError,
    UnsupportedInput,
    WeightsReceipt,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
    canonical_json,
)
from h3_tables import job
from h3_tables._table_layout import TableLayout
from h3_tables.adaln_operations import _bank_config, _validate_body_config
from h3_tables.kernel import H3Topology, table_shapes
from h3_tables.legacy_config import upgrade_legacy_table_config
from h3_tables.model_config import dual_adaln_pruned_config, parse_production_config
from h3_tables.plans import TASKS


class Recorder:
    """Only config/structure access is available: any weight read fails this proof."""

    def __init__(self, raw: bytes, shapes_from: bytes) -> None:
        self.raw = raw
        self.opened = False
        self.writes: list[bytes] = []
        self.replayed = False
        self.receipt = None
        self.targets: dict[str, Any] = {}
        document = canonical_json.decode(shapes_from)
        rows = []
        for task in TASKS:
            component = f"{task}_dit"
            layout = TableLayout.parse(document[component]["cozy_h3"]["table_keys"])
            plan = replace(
                job._production_plan(task), timesteps=layout.timesteps, block_rows=layout.block_keys
            )
            for key, shape in table_shapes(
                H3Topology.from_config(document[component]), plan
            ).items():
                rows.append(self.tensor(component, key, "bf16", shape))
        rows.extend(
            [
                self.tensor("video_vae", "decoder.proj_in.weight", "f16", (2, 2)),
                self.tensor("text_encoder", "weight", "bf16", (2, 2)),
                self.tensor("audio_vae", "weight", "bf16", (2, 2)),
            ]
        )
        self.source = WeightsSource(("model",), tuple(rows))

    @staticmethod
    def tensor(component: str, key: str, dtype: str, shape: tuple[int, ...]) -> WeightsSourceTensor:
        return WeightsSourceTensor(
            component,
            key,
            dtype,
            shape,
            (WeightsSourcePart("value", dtype, shape),),
            encoding=job.PLAIN_SPEC,
        )

    def structure(self, _: Any) -> WeightsSource:
        return self.source

    def config(self, _: Any, name: str) -> bytes:
        assert name == "model"
        return self.raw

    def open(self, slot: str, **kwargs: Any) -> Any:
        assert slot == "restamped"
        self.opened = True
        self.targets = kwargs["targets"]
        return nullcontext(self)

    def add_config(self, name: str, raw: bytes) -> None:
        assert name == "model"
        self.writes.append(raw)

    def commit(self) -> WeightsReceipt:
        assert len(self.writes) == 1
        return WeightsReceipt("proof", "proof", "proof", b"")

    def metric(self, *_: Any, **__: Any) -> None:
        pass


def refuses(action: Callable[[], Any]) -> None:
    try:
        action()
    except (ConformanceError, UnsupportedInput, ValueError):
        return
    raise AssertionError("invalid metadata was accepted")


def main() -> None:
    plans = job._plans(job.LAUNCH_SET)
    current = job._asset("model-config.json")
    sections = parse_production_config(current)
    assert dual_adaln_pruned_config(sections, plans["fl2va"], plans["ref2va"]) == current
    legacy = canonical_json.decode(current)
    for task, plan in plans.items():
        document = canonical_json.decode(plan.canonical_bytes)
        document["frames"] = 345
        stamp = legacy[f"{task}_dit"]["cozy_h3"]
        del stamp["table_keys"]
        stamp["timestep_plan_digest"] = canonical_json.digest(document)
    legacy_raw = canonical_json.encode(legacy)
    assert upgrade_legacy_table_config(legacy_raw, plans) == current
    assert upgrade_legacy_table_config(current, plans) == current

    # The historical digest proof fails if the producer's row meanings or schedule changed.
    for change in ("keys", "schedule"):
        plan = plans["fl2va"]
        changed = canonical_json.decode(plan.canonical_bytes)
        if change == "keys":
            changed["table_keys"]["final_normalization"][0]["timestep"] = (0.5).hex()
        else:
            changed["schedules"][0]["evaluations"][1]["video_sigma"] = (0.5).hex()
        altered = replace(plan, canonical_bytes=canonical_json.encode(changed))
        refuses(partial(upgrade_legacy_table_config, legacy_raw, {**plans, "fl2va": altered}))

    # Different valid row order is preserved; the restamp must not relabel existing bytes.
    custom = canonical_json.decode(current)
    for task in TASKS:
        stamp = custom[f"{task}_dit"]["cozy_h3"]
        stamp["generating_projection_digest"] = "sha256:" + "ab" * 32
        for rows in stamp["table_keys"].values():
            rows.reverse()
            for index, row in enumerate(rows):
                row["index"] = index
    custom_raw = canonical_json.encode(custom)
    assert upgrade_legacy_table_config(custom_raw, plans) == custom_raw
    for raw in (legacy_raw, current, custom_raw):
        expected = custom_raw if raw == custom_raw else current
        recorder = Recorder(raw, expected)
        result = job.restamp(recorder, job.ProductionRequest(), None, recorder, recorder)
        assert recorder.writes == [expected]
        assert result.source_bytes_read == result.new_bytes_written == result.cast_keys == 0
        assert result.normalised_components == ()
        assert all(not target.add and not target.drop for target in recorder.targets.values())

    # Counts, duplicate row labels, and unknown legacy origins fail before a write opens.
    unknown = copy.deepcopy(legacy)
    unknown["fl2va_dit"]["cozy_h3"]["timestep_plan_digest"] = "sha256:" + "ff" * 32
    duplicate = canonical_json.decode(current)
    rows = duplicate["fl2va_dit"]["cozy_h3"]["table_keys"]["final_normalization"]
    rows[1]["timestep"] = rows[0]["timestep"]
    for document in (unknown, duplicate):
        recorder = Recorder(canonical_json.encode(document), current)
        refuses(partial(job.restamp, recorder, job.ProductionRequest(), None, recorder, recorder))
        assert not recorder.opened and not recorder.writes
    recorder = Recorder(current, current)
    first = recorder.source.tensors[0]
    recorder.source = replace(
        recorder.source, tensors=(replace(first, shape=(1, 6, 5376)), *recorder.source.tensors[1:])
    )
    refuses(lambda: job.restamp(recorder, job.ProductionRequest(), None, recorder, recorder))
    assert not recorder.opened

    # Body and bank selection use row layouts, not formatting or full plan identities.
    layouts = _validate_body_config(Recorder(custom_raw, custom_raw), None, pruned=True)
    assert layouts["fl2va"] == custom["fl2va_dit"]["cozy_h3"]["table_keys"]
    bank = canonical_json.decode(_bank_config("fl2va", "sha256:" + "11" * 32))
    assert "plan" not in bank and bank["table_keys"] == plans["fl2va"].table_keys
    print(
        "PASS legacy provenance + layout proof; arbitrary row order preserved; "
        "no-cast config write; pre-write refusals"
    )


if __name__ == "__main__":
    main()
