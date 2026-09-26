"""Derive a private 30-step coverage fixture; never changes the production plan pin."""

import argparse
import json
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

from cozy_runtime import canonical_json
from h3_tables.plans import parse_plan


class BlockRow(TypedDict):
    index: int
    timestep: str
    modality: str
    modality_tag: int


class FinalRow(TypedDict):
    index: int
    timestep: str


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("fl2va", "ref2va"), default="fl2va")
parser.add_argument(
    "--out", type=Path, default=Path.home() / ".cozy/outputs/h3-bank-qualification/coverage-30.json"
)
args = parser.parse_args()
document = json.loads(
    files("h3_tables").joinpath("assets", f"timestep-plan.{args.task}.json").read_bytes()
)
document["schedules"] = [
    schedule for schedule in document["schedules"] if schedule["transformer_evaluations"] == 30
]
if len(document["schedules"]) != 1:
    raise SystemExit("the source must contain exactly one reviewed 30-step schedule")
blocks: list[BlockRow] = []
final: list[FinalRow] = []
seen_blocks: set[tuple[str, int]] = set()
seen_final: set[str] = set()
for evaluation in document["schedules"][0]["evaluations"]:
    for item in evaluation["modulation_classes"]:
        pair = (item["timestep"], item["modality_tag"])
        if pair not in seen_blocks:
            seen_blocks.add(pair)
            blocks.append(
                {
                    "index": len(blocks),
                    "timestep": pair[0],
                    "modality": item["modality"],
                    "modality_tag": pair[1],
                }
            )
        if pair[0] not in seen_final:
            seen_final.add(pair[0])
            final.append({"index": len(final), "timestep": pair[0]})
document["table_keys"] = {"block_modulation": blocks, "final_normalization": final}
raw = canonical_json.encode(document)
plan = parse_plan(raw, task=args.task, launch=False)
try:
    parse_plan(raw, task=args.task)
except ValueError as error:
    if "not the launch oracle" not in str(error):
        raise
else:
    raise SystemExit("qualification variant unexpectedly equals the production launch plan")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_bytes(raw)
print(
    json.dumps(
        {
            "task": args.task,
            "digest": plan.digest,
            "steps": plan.steps,
            "timesteps": len(plan.timesteps),
            "block_rows": len(plan.block_rows),
            "file": str(args.out),
            "production_launch_refused": True,
        },
        sort_keys=True,
    )
)
