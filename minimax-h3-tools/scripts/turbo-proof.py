#!/usr/bin/env python3
"""PDD-8 turbo tables on CPU: the plan, the fused kernel, the declarations, the orchestration.

No GPU, no store, no weight byte. The grid functions are transcribed verbatim from
`alibaba-pai/MiniMax-H3-Acc-LoRAs` `minimax_h3_pdd.py` (revision
335001fb9e5455d68a0caa18ec2e319072150328, sha256
7a4c04bd364f84c1e6dfecf816bdd991eb7092e3c69693fea64b8e6db778e4c9), as is `LoRALinear.forward`.
"""

from __future__ import annotations

import copy
import json
import struct
from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import torch
from cozy_runtime.author import (
    WeightsReceipt,
    WeightsSource,
    WeightsSourcePart,
    WeightsSourceTensor,
    canonical_json,
)
from h3_tables import job
from h3_tables.kernel import (
    H3Topology,
    LowRankAdapter,
    _timestep_features,
    adapter_shapes,
    precompute_tables,
    removed_keys,
    source_shapes,
    table_bytes,
    table_shapes,
)
from h3_tables.model_config import dual_adaln_pruned_config
from h3_tables.plans import (
    AUDIO_SHIFT,
    LAUNCH_GRID_POINTS,
    LAUNCH_PLAN_DIGESTS,
    TURBO_GRID_POINTS,
    TURBO_PLAN_DIGESTS,
    VIDEO_SHIFT,
    compose_plan,
    parse_declared_plan,
    parse_plan,
    sigma_grid,
)
from h3_tables.source import (
    ADAPTER_ALPHA,
    ADAPTER_RANK,
    H3FullTransformer,
    H3TurboAdapter,
    adapter_slice,
)
from torch.nn import functional as F

TASKS = ("fl2va", "ref2va")
PDD_NUM_STEPS = 32
PDD_BLOCK_SIZE = 4
TINY = H3Topology(8, 3, 8, 12, 4)


def f32(value: float) -> float:
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def shifted_sigma(shift: float, sigma: torch.Tensor) -> torch.Tensor:
    return shift * sigma / (1 + (shift - 1) * sigma)


def pdd_time_grid(shift: float, num_steps: int) -> torch.Tensor:
    """Ascending grid `0 = t_0 < ... < t_N = 1` of one MiniMax-H3 schedule."""
    sigma = torch.linspace(1.0, 0.0, num_steps + 1, dtype=torch.float64)
    return 1.0 - shifted_sigma(shift, sigma)


def reference_lora_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    lora_down: torch.Tensor,
    lora_up: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    out = F.linear(x, weight, bias)
    update = F.linear(
        F.linear(x, lora_down.to(device=x.device, dtype=x.dtype)),
        lora_up.to(device=x.device, dtype=x.dtype),
    )
    return out + scaling * update.to(out.dtype)


def fail(what: str) -> None:
    raise SystemExit(f"turbo-proof: {what}")


def refuses(what: str, action: Any) -> None:
    try:
        action()
    except ValueError:
        return
    fail(f"{what} did not refuse")


def prove_plans() -> dict[str, Any]:
    assets = job.files("h3_tables").joinpath("assets")
    plans: dict[str, Any] = {}
    for task in TASKS:
        launch_raw = assets.joinpath(f"timestep-plan.{task}.json").read_bytes()
        turbo_raw = assets.joinpath(f"timestep-plan.{task}.turbo.json").read_bytes()
        if compose_plan(task, grid_points=LAUNCH_GRID_POINTS) != launch_raw:
            fail(f"{task} launch plan does not reproduce from the composer")
        if compose_plan(task, grid_points=TURBO_GRID_POINTS) != turbo_raw:
            fail(f"{task} turbo plan does not reproduce from the composer")
        turbo = parse_declared_plan(turbo_raw, digests=TURBO_PLAN_DIGESTS)
        if turbo.steps != (8,) or len(turbo.timesteps) != 17 or len(turbo.block_rows) != 26:
            fail(f"{task} turbo plan is {turbo.steps} steps, {len(turbo.block_rows)} block rows")
        refuses(
            f"{task} turbo bytes under the launch pin",
            lambda raw=turbo_raw: parse_declared_plan(raw),
        )
        refuses(
            f"{task} launch bytes under the turbo pin",
            lambda raw=launch_raw: parse_declared_plan(raw, digests=TURBO_PLAN_DIGESTS),
        )
        changed = copy.deepcopy(json.loads(turbo_raw))
        changed["schedules"][0]["evaluations"][1]["video_sigma"] = (0.5).hex()
        altered = canonical_json.encode(changed)
        refuses(
            f"a changed {task} turbo plan",
            lambda raw=altered, name=task: parse_plan(raw, task=name, digests=TURBO_PLAN_DIGESTS),
        )
        plans[task] = turbo
        document = json.loads(turbo_raw)
        schedule = document["schedules"][0]
        if (
            schedule["transformer_evaluations"] != PDD_NUM_STEPS // PDD_BLOCK_SIZE
            or schedule["sigma_grid_points"] != PDD_NUM_STEPS // PDD_BLOCK_SIZE + 1
            or float.fromhex(document["video_shift"]) != VIDEO_SHIFT
            or float.fromhex(document["audio_shift"]) != AUDIO_SHIFT
        ):
            fail(f"{task} turbo schedule is not eight evaluations over nine grid points")
        for modality, shift in (("video", VIDEO_SHIFT), ("audio", AUDIO_SHIFT)):
            ours = sigma_grid(shift, PDD_NUM_STEPS // PDD_BLOCK_SIZE + 1)
            declared = (
                *(float.fromhex(row[f"{modality}_sigma"]) for row in schedule["evaluations"]),
                float.fromhex(schedule["terminal"][f"{modality}_sigma"]),
            )
            if declared != ours:
                fail(f"{task} {modality} sigmas are not the scheduler's own float32 grid")
            training = (1.0 - pdd_time_grid(shift, PDD_NUM_STEPS))[::PDD_BLOCK_SIZE]
            if tuple(f32(float(value)) for value in training) != ours:
                fail(f"{task} {modality} grid differs from PDD's block boundaries in float32")
            served = tuple(f32(1.0 - sigma) for sigma in ours[:-1])
            grid = pdd_time_grid(shift, PDD_NUM_STEPS)[::PDD_BLOCK_SIZE][:-1]
            rounded_once = tuple(f32(float(t)) for t in grid)
            off = sum(a != b for a, b in zip(served, rounded_once, strict=True))
            print(
                f"  {task} {modality}: 9 sigmas equal PDD's float64 grid at float32; served "
                f"timesteps 1-sigma differ from f32(1-sigma64) at {off}/8 points by double "
                "rounding (the reference feeds the scheduler's own timesteps, never its "
                "float64 grid)"
            )
    return plans


def prove_budget(sections: dict[str, dict[str, Any]], turbo: dict[str, Any]) -> None:
    topology = H3Topology.from_config(sections["transformer"])
    launch = parse_declared_plan(job._asset("timestep-plan.fl2va.json"))
    turbo_bytes = table_bytes(topology, turbo["fl2va"])
    expected = 50 * 26 * 6 * 5376 * 2 + 17 * 2 * 5376 * 2
    launch_bytes = table_bytes(topology, launch)
    if turbo_bytes != expected or turbo_bytes != 84_231_168:
        fail(f"turbo tables are {turbo_bytes} bytes, expected {expected}")
    if launch_bytes != 1_020_515_328 or launch_bytes > job.MAX_TABLE_BYTES:
        fail(f"launch tables are {launch_bytes} bytes")
    job._check_table_budget(job.MAX_TABLE_BYTES)
    refuses("a budget below the launch plan", lambda: job._check_table_budget(launch_bytes - 1))
    print(
        f"  table bytes per task: launch {launch_bytes} ({launch_bytes / job.MAX_TABLE_BYTES:.1%}),"
        f" turbo {turbo_bytes} ({turbo_bytes / job.MAX_TABLE_BYTES:.1%}) of {job.MAX_TABLE_BYTES}"
    )
    raw = dual_adaln_pruned_config(sections, turbo["fl2va"], turbo["ref2va"])
    document = canonical_json.decode(raw)
    if any(
        document[f"{task}_dit"]["cozy_h3"]["table_keys"] != turbo[task].table_keys
        or "timestep_plan_digest" in document[f"{task}_dit"]["cozy_h3"]
        for task in TASKS
    ):
        fail("the turbo model config does not describe its actual table rows")
    launch_ref = parse_declared_plan(job._asset("timestep-plan.ref2va.json"))
    mixed = canonical_json.decode(dual_adaln_pruned_config(sections, turbo["fl2va"], launch_ref))
    if (
        mixed["fl2va_dit"]["cozy_h3"]["table_keys"] != turbo["fl2va"].table_keys
        or mixed["ref2va_dit"]["cozy_h3"]["table_keys"] != launch_ref.table_keys
    ):
        fail("independent task tables lost their own row meanings")
    committed = canonical_json.decode(job._asset("model-config.json"))
    rebuilt = canonical_json.decode(dual_adaln_pruned_config(sections, launch, launch_ref))
    if canonical_json.digest(committed) != canonical_json.digest(rebuilt):
        fail("the launch config no longer reproduces the committed asset")


def prove_kernel(turbo: dict[str, Any]) -> None:
    plan = turbo["fl2va"]
    torch.manual_seed(31)
    rank = 4
    base = {
        key: (torch.randn(shape) * 0.05).to(dtype)
        for key, (dtype, shape) in source_shapes(TINY).items()
    }
    lora = {
        key: (torch.randn(shape) * 0.2).to(dtype)
        for key, (dtype, shape) in adapter_shapes(TINY, rank).items()
    }
    lora["transformer_blocks.0.attn.to_q.lora_down"] = torch.randn(rank, 8).bfloat16()
    base_reads: list[str] = []
    adapter_reads: list[str] = []

    def read_base(key: str, _dtype: torch.dtype, _shape: tuple[int, ...]) -> torch.Tensor:
        base_reads.append(key)
        return base[key]

    def read_adapter(key: str, _dtype: torch.dtype, _shape: tuple[int, ...]) -> torch.Tensor:
        adapter_reads.append(key)
        return lora[key]

    def run(adapter: LowRankAdapter | None) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        precompute_tables(
            plan=plan,
            topology=TINY,
            read=read_base,
            write=lambda key, value: out.__setitem__(key, value.clone()),
            progress=lambda _done, _total: None,
            device=torch.device("cpu"),
            adapter=adapter,
        )
        return out

    plain = run(None)
    plain_reads = list(base_reads)
    base_reads.clear()
    scale = ADAPTER_ALPHA / ADAPTER_RANK
    fused = run(LowRankAdapter(rank, scale, read_adapter))
    if base_reads != plain_reads:
        fail("the fused pass changed which modulation weights it reads")
    if set(adapter_reads) != set(adapter_shapes(TINY, rank)) or len(adapter_reads) != 2 * len(
        range(TINY.num_layers)
    ):
        fail(f"the fused pass read {sorted(adapter_reads)} from the adapter")
    if set(fused) != set(plain) != set(table_shapes(TINY, plan)):
        fail("fused and plain passes emit different tables")
    if not torch.equal(fused["norm_out.table"], plain["norm_out.table"]):
        fail("the final-normalization table moved under an adapter that does not target it")

    timestep = torch.tensor(plan.timesteps, dtype=torch.float32)
    temb = F.linear(
        F.silu(
            F.linear(
                _timestep_features(timestep, TINY.freq_dim),
                base["time_embedder.linear_1.weight"],
                base["time_embedder.linear_1.bias"],
            )
        ),
        base["time_embedder.linear_2.weight"],
        base["time_embedder.linear_2.bias"],
    )
    activated = F.silu(temb).to(torch.bfloat16)
    rows = torch.tensor([row * 3 + modality for row, modality in plan.block_rows])
    for index in range(TINY.num_layers):
        prefix = f"transformer_blocks.{index}.adaln_proj.linear"
        expected = reference_lora_forward(
            activated,
            base[f"{prefix}.weight"],
            base[f"{prefix}.bias"],
            lora[f"{prefix}.lora_down"],
            lora[f"{prefix}.lora_up"],
            scale,
        ).reshape(-1, 6, TINY.hidden_size)[rows]
        table = fused[f"transformer_blocks.{index}.adaln_proj.table"]
        if not torch.equal(table, expected):
            fail(f"block {index} fused table is not the reference LoRALinear output at plan rows")
        if torch.equal(table, plain[f"transformer_blocks.{index}.adaln_proj.table"]):
            fail(f"block {index} fused table equals the unfused table")
        term = (
            scale
            * F.linear(F.linear(activated, lora[f"{prefix}.lora_down"]), lora[f"{prefix}.lora_up"])
        ).reshape(-1, 6, TINY.hidden_size)[rows]
        delta = table.float() - plain[f"transformer_blocks.{index}.adaln_proj.table"].float()
        if not torch.allclose(delta, term.float(), atol=2**-6, rtol=2**-7):
            fail(f"block {index} fused - unfused is not the LoRA term to bf16 rounding")
    print(
        f"  kernel: {TINY.num_layers} block tables equal the reference LoRALinear output at every "
        f"plan row (exact); norm_out untouched; adapter reads = the adaln slice only"
    )


def structure(rows: list[tuple[str, str, str, tuple[int, ...]]]) -> WeightsSource:
    return WeightsSource(
        ("model",),
        tuple(
            WeightsSourceTensor(
                component, key, dtype, shape, (WeightsSourcePart("value", dtype, shape),)
            )
            for component, key, dtype, shape in rows
        ),
    )


class Interrupted(RuntimeError):
    pass


class Transaction:
    def __init__(self, owner: Recorder, slot: str, sources: Any, order: Any) -> None:
        self.owner, self.slot = owner, slot
        self.sources, self.order = tuple(sources), tuple(order)
        self.replayed = slot in owner.committed
        prior = owner.committed.get(slot)
        self.receipt = replace(prior, replayed=True) if prior else None
        self.config: bytes | None = None

    def __enter__(self) -> Transaction:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def add_config(self, name: str, raw: bytes) -> None:
        assert not self.replayed and name == "model" and raw
        self.config = raw

    def commit(self) -> WeightsReceipt:
        assert not self.replayed and self.config is not None
        assert self.slot not in self.owner.committed
        receipt = WeightsReceipt(self.slot, self.slot, self.slot, b"")
        self.owner.committed[self.slot] = receipt
        self.owner.configs[self.slot] = self.config
        self.owner.events.append("commit:" + self.slot)
        return receipt


class Recorder:
    def __init__(self, structures: dict[str, WeightsSource]) -> None:
        self.structures = structures
        self.committed: dict[str, WeightsReceipt] = {}
        self.configs: dict[str, bytes] = {}
        self.events: list[str] = []
        self.opened: dict[str, Transaction] = {}
        self.fail_at = ""

    def structure(self, model: Any) -> WeightsSource:
        return self.structures[model.checkpoint_ref]

    def open(
        self, slot: str, *, sources: Any, targets: Any, configs: Any, order: Any
    ) -> Transaction:
        del targets, configs
        transaction = Transaction(self, slot, sources, order)
        self.opened[slot] = transaction
        return transaction

    def tables(
        self,
        task: str,
        plan: Any,
        _topology: Any,
        _ctx: Any,
        _source: Any,
        active: dict[str, Any],
        _tel: Any,
        _range: Any,
        *,
        source: str,
        adapter: Any,
        device: Any,
    ) -> tuple[int, int]:
        name = "turbo" if plan.steps == (8,) else "launch"
        event = f"tables:{name}:{task}"
        self.events.append(f"{event}:{source}:{adapter}:{device.type}:{','.join(sorted(active))}")
        if self.fail_at == event:
            raise Interrupted(event)
        return 1, 1


class Telemetry:
    def stage(self, *_: Any, **__: Any) -> Any:
        return nullcontext()

    def progress(self, *_: Any, **__: Any) -> None:
        pass

    def metric(self, *_: Any, **__: Any) -> None:
        pass

    def log(self, *_: Any, **__: Any) -> None:
        pass


def structures(sections: dict[str, dict[str, Any]]) -> dict[str, WeightsSource]:
    topologies = job._topologies(sections)
    launch = job._table_additions(topologies, job._plans(job.LAUNCH_SET))
    shared = [(c, f"{c}.w", "f32", (1,)) for c in ("text_encoder", "video_vae", "audio_vae")]
    dtype_of = {"torch.float32": "f32", "torch.bfloat16": "bf16"}
    full_rows, pruned_rows = list(shared), list(shared)
    adapters: dict[str, WeightsSource] = {}
    for task in TASKS:
        component = f"{task}_dit"
        full_rows += [(component, "proj_in.weight", "bf16", (2, 2))]
        full_rows += [
            (component, key, dtype_of[str(dtype)], shape)
            for key, (dtype, shape) in source_shapes(topologies[task]).items()
        ]
        pruned_rows += [(component, "proj_in.weight", "bf16", (2, 2))]
        pruned_rows += [
            (component, key, "bf16", tensor.shape) for key, tensor in launch[task].items()
        ]
        rows = [("model", "proj_out.weight", "bf16", (32, 64, 5376))]
        rows += [
            ("model", key, "bf16", shape)
            for key, (_, shape) in adapter_shapes(topologies[task], ADAPTER_RANK).items()
        ]
        adapters[f"test://{task}-adapter"] = structure(rows)
    return {
        "test://full": structure(full_rows),
        "test://pruned": structure(pruned_rows),
        **adapters,
    }


def invoke(recorder: Recorder, fail_at: str = "") -> Any:
    recorder.fail_at = fail_at
    module: Any = job
    original_tables = job._write_tables
    original_cuda = job.torch.cuda.is_available
    original_tf32 = job.torch.backends.cuda.matmul.allow_tf32
    original_precision = job.torch.get_float32_matmul_precision()
    try:
        module._write_tables = recorder.tables
        module.torch.cuda.is_available = lambda: True
        return module.retable(
            None,
            job.ProductionRequest(),
            H3FullTransformer.for_test(checkpoint_ref="test://full"),
            H3FullTransformer.for_test(checkpoint_ref="test://pruned"),
            H3TurboAdapter.for_test(checkpoint_ref="test://fl2va-adapter"),
            H3TurboAdapter.for_test(checkpoint_ref="test://ref2va-adapter"),
            recorder,
            Telemetry(),
        )
    finally:
        module._write_tables = original_tables
        module.torch.cuda.is_available = original_cuda
        job.torch.set_float32_matmul_precision(original_precision)
        job.torch.backends.cuda.matmul.allow_tf32 = original_tf32


def prove_orchestration(sections: dict[str, dict[str, Any]]) -> None:
    slots = ("adaln-pruned", "tables", "turbo-adaln-pruned", "turbo-tables")
    recorder = Recorder(structures(sections))
    try:
        invoke(recorder, "tables:turbo:fl2va")
    except Interrupted:
        pass
    else:
        fail("the injected turbo interruption did not fire")
    if tuple(recorder.committed) != slots[:2]:
        fail(f"a turbo interruption retained {tuple(recorder.committed)}")
    expected_events = [
        "tables:launch:fl2va:full:None:cuda:bank,checkpoint",
        "tables:launch:ref2va:full:None:cuda:bank,checkpoint",
        "commit:adaln-pruned",
        "commit:tables",
        "tables:turbo:fl2va:full:('fl2va_adapter', 'model'):cuda:bank,checkpoint",
    ]
    if recorder.events != expected_events:
        fail(f"first attempt ran {recorder.events}")
    if set(recorder.opened["tables"].sources) != {"full"} or set(
        recorder.opened["turbo-tables"].sources
    ) != {"full", "fl2va_adapter", "ref2va_adapter"}:
        fail("the banks do not name exactly the sources their targets derive from")
    turbo_order = recorder.opened["turbo-tables"].order
    slice_keys = list(adapter_shapes(job._topologies(sections)["fl2va"], ADAPTER_RANK))
    if (
        turbo_order[:102] != tuple(recorder.opened["tables"].order)
        or list(turbo_order[102:202]) != [("fl2va_adapter", key) for key in slice_keys]
        or list(turbo_order[202:]) != [("ref2va_adapter", key) for key in slice_keys]
    ):
        fail("the turbo bank order is not the table rows then each adapter's slice")
    if recorder.configs["adaln-pruned"] != job._asset("model-config.json"):
        fail("the launch checkpoint config is not the committed model config")

    recorder.events.clear()
    result = invoke(recorder)
    if tuple(recorder.committed) != slots:
        fail(f"the resumed attempt retained {tuple(recorder.committed)}")
    if recorder.events != [
        "tables:turbo:fl2va:full:('fl2va_adapter', 'model'):cuda:bank,checkpoint",
        "tables:turbo:ref2va:full:('ref2va_adapter', 'model'):cuda:bank,checkpoint",
        "commit:turbo-adaln-pruned",
        "commit:turbo-tables",
    ]:
        fail(f"the resumed attempt ran {recorder.events}")
    turbo_config = canonical_json.decode(recorder.configs["turbo-adaln-pruned"])
    if any(
        turbo_config[f"{task}_dit"]["cozy_h3"]["table_keys"]
        != job._production_plan(task, job.TABLE_SETS[1]).table_keys
        for task in TASKS
    ):
        fail("the turbo checkpoint config does not describe its actual table rows")
    by_name = {row.table_set: row for row in result.table_sets}
    if (
        [row.table_set for row in result.table_sets] != ["launch", "turbo"]
        or by_name["launch"].steps != [30, 40, 50]
        or by_name["turbo"].steps != [8]
        or by_name["launch"].plan_digests != dict(LAUNCH_PLAN_DIGESTS)
        or by_name["turbo"].plan_digests != dict(TURBO_PLAN_DIGESTS)
        or by_name["launch"].adapters != []
        or by_name["turbo"].adapters != ["test://fl2va-adapter", "test://ref2va-adapter"]
        or not by_name["launch"].replayed
        or by_name["turbo"].replayed
        or by_name["launch"].table_bytes_this_run != 0
        or by_name["turbo"].table_bytes_this_run != 2
        or any(row.table_tensors != 102 for row in result.table_sets)
        or result.source_checkpoint != "test://pruned"
        or result.source_bytes_read_this_run != 2
    ):
        fail(f"the resumed result is {result}")

    recorder.events.clear()
    replay = invoke(recorder)
    if recorder.events or not all(row.replayed for row in replay.table_sets):
        fail(f"a full replay ran {recorder.events}")
    if replay.source_bytes_read_this_run != 0:
        fail("a full replay read source bytes")
    print("  orchestration: launch then turbo, checkpoint before bank, interruption and replay")


def prove_adapter_refusals(sections: dict[str, dict[str, Any]]) -> None:
    topology = job._topologies(sections)["fl2va"]
    good = structures(sections)["test://fl2va-adapter"]
    component, unread = adapter_slice(good, topology)
    if component != "model" or unread != ("proj_out.weight",):
        fail(f"the adapter slice resolved to {component}/{unread}")
    if set(removed_keys(topology)) & set(adapter_shapes(topology, ADAPTER_RANK)):
        fail("adapter keys collide with the dropped modulation keys")


def main() -> None:
    sections = job.parse_production_config(job._asset("model-config.json"))
    turbo = prove_plans()
    prove_budget(sections, turbo)
    prove_kernel(turbo)
    prove_adapter_refusals(sections)
    prove_orchestration(sections)
    print(
        "H3 TURBO PROOF PASS plans=composer-exact grid=pdd-block-boundaries "
        "kernel=reference-exact budget=unchanged outputs=4 replay=ok"
    )


if __name__ == "__main__":
    main()
