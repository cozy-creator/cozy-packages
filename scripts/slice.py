#!/usr/bin/env python
"""The local transport: ONE endpoint request through the real cozy-runtime path.

`cozy_runtime.internal.local.run_slice` is cr-008a's door — "a real endpoint request with
no coordinator, no hub, no gRPC" — and this module is the thin thing that turns it into a
`cozy_eval.wire.Transport`. Everything below the in-memory control adapter is the
production path: the same acceptance boundary, the same durable journal, the same
supervisor/executor process split, the same ledger and output transaction.

Two processes, on purpose, exactly as in production:

    the CALLER   an eval job: numpy, PIL, ffmpeg, cozy-eval. Never imports torch.
    the WORKER   `python -m scripts.slice exec <spec>`: the supervisor, its disposable
                 CUDA executor child, and the endpoint module.

So `SliceTransport.invoke` spawns a process, and a killed run is a real killed run. Each
invocation is a cold one-shot: the executor is spawned, the artifact is filled, the request
is served and the process dies. That is why the endpoint's requests are BATCH-shaped — a
4.9 GiB fill per question would be the whole cost of the pass.

ONE binding plan is staged per invocation, never two. cr-008a's executor keeps
`self.models` for a SINGLE prepared binding, so staging both the judge and the transcriber
would leave whichever prepared last and silently answer the other entrypoint with the wrong
model. Multi-binding residency in one executor is cr-008b's; the seam is named on ev-003.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "quality-judge"
ARTIFACTS = Path.home() / "cozy_v2" / "eval-models" / "artifacts"
WORKSPACES = Path("/tmp/cozy-judge-slices")
MEDIA = WORKSPACES / "media"
SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png", "audio/x-f32le": ".f32"}

#: Which model each entrypoint binds. The endpoint declares this in its signatures; the
#: driver needs it to stage the one right binding plan, and reading it off the committed
#: descriptor rather than restating it keeps the two from drifting.
BINDINGS = {"judge": "judge", "soft": "judge", "pairwise": "judge", "transcribe": "transcriber"}


def descriptor_bindings() -> dict[str, str]:
    """entrypoint -> model CLASS, from the committed endpoint descriptor."""
    document = json.loads((PROJECT / "endpoint.descriptor.json").read_text())
    return {
        str(entry["name"]): str(entry["models"][0]["class"])
        for entry in document["entrypoints"]
        if entry.get("models")
    }


def binding_record(entrypoint: str, family: str) -> dict[str, Any]:
    record = json.loads((ARTIFACTS / family / "binding.json").read_text())
    plan = "sha256:" + hashlib.sha256(
        f"ebp/{record['release']}/{entrypoint}".encode()
    ).hexdigest()
    return {
        "plan_id": plan,
        "project": str(PROJECT),
        "model_class": record["model_class"],
        "binding_path": f"{entrypoint}.models.model",
        "param": "model",
        "component": record["component"],
        "store": record["store"],
        "config": record["config"],
        "snapshot": record["snapshot"],
        "release": record["release"],
        "variant": record["variant"],
        "vram_bytes": record["vram_bytes"],
        "host_bytes": record["host_bytes"],
        "pinned_bytes": record["pinned_bytes"],
        "entrypoint": entrypoint,
        "model_construction_digest": "",
    }


class _Shortfall(RuntimeError):
    """The card had no room for this fill. A resource condition, never an answer."""


def _needed_mib(entrypoint: str) -> int:
    record = binding_record(entrypoint, BINDINGS[entrypoint])
    return int(record["vram_bytes"] / (1 << 20)) + 256


@dataclass
class SliceTransport:
    """A `cozy_eval.wire.Transport` backed by one local run per invocation."""

    deadline_seconds: float = 3600.0
    verbose: bool = True
    #: What one request document may weigh. cozy-runtime's supervisor/executor seam is
    #: CONTROL: `internal/seam.py` caps a frame at 64 KiB precisely so "a caller trying to
    #: move tensors or media through here fails immediately". A request over it does NOT
    #: fail immediately here, though — measured: a 73 721 B soft batch (108 calls, 432
    #: asset refs) HANGS the local run, because the supervisor-side `SeamError` escapes
    #: after the attempt was accepted, no terminal is ever produced, and `LocalCoordinator`
    #: only deadlines a run that never dispatched. Recorded on ev-003 as a cr-008a finding;
    #: the `timeout=` below is what makes it visible instead of eternal.
    max_payload_bytes: int = 56 * 1024
    #: A slice that produces no terminal must not wait forever.
    worker_timeout: float = 900.0
    #: The WORKER interpreter — the one with torch, transformers and the tensorfs facade.
    #: Named rather than inherited from `sys.executable` because the caller is an eval job
    #: and has no business being the same environment as a CUDA executor's parent.
    worker_python: str = str(ROOT / ".venv" / "bin" / "python")
    #: Every run's observations, in order — timings, ledger, terminal, exit code.
    runs: list[dict[str, Any]] = field(default_factory=list)
    staged_bytes: int = 0

    def stage(self, data: bytes, media_type: str) -> str:
        """Write one blob where the worker can read it, and return its asset ref.

        THE COORDINATOR'S JOB, locally. On the network path an eval job uploads media and
        the hub grants the worker a URL for it; here the same handle is a `file://` ref the
        kernel hydrates into the attempt spool (`author._invoke._hydrate`). The endpoint
        cannot tell the difference and must not be able to: it receives a typed asset and
        reads verified bytes, never a path it chose.

        Content-addressed, so re-staging the same frame is free and two comparisons that
        share a strip share one blob.
        """
        MEDIA.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(data).hexdigest()
        path = MEDIA / f"{digest}{SUFFIX.get(media_type, '.bin')}"
        if not path.is_file():
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        self.staged_bytes += len(data)
        return f"file://{path}"

    def invoke(self, entrypoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """One request. A DEVICE SHORTFALL is retried once; nothing else is.

        This card is shared with other agents, so another resident can take the 4.9 GiB
        this fill needs between the moment the driver checks for room and the moment the
        executor asks for it. The runtime reports that correctly, as a typed
        `device_shortfall` fault with both numbers in it — it is a resource condition, not
        a result, and retrying it once after waiting is the honest response. Every other
        failure raises: a refusal is an ANSWER and must never be retried into a different
        one.
        """
        try:
            return self._invoke(entrypoint, payload)
        except _Shortfall as first:
            print(f"   [slice] {entrypoint}: {first}; waiting for the card", flush=True)
            _settle(_needed_mib(entrypoint))
            return self._invoke(entrypoint, payload)

    def _invoke(self, entrypoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        family = BINDINGS[entrypoint]
        stamp = f"{entrypoint}-{len(self.runs)}-{int(time.time())}"
        workspace = WORKSPACES / stamp
        workspace.mkdir(parents=True, exist_ok=True)
        spec = {
            "entrypoint": entrypoint,
            "payload": payload,
            "binding": binding_record(entrypoint, family),
            "workspace": str(workspace),
            "release_id": binding_record(entrypoint, family)["release"],
            "request_id": stamp,
            "deadline_ms": int((time.time() + self.deadline_seconds) * 1000),
        }
        spec_file = workspace / "spec.json"
        spec_file.write_text(json.dumps(spec))
        out_file = workspace / "outcome.json"
        need = _needed_mib(entrypoint)
        waited = _settle(need)
        if waited > 1.0 and self.verbose:
            print(f"   [slice] waited {waited:.0f}s for {need} MiB of card", flush=True)
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                ["nice", "-n", "19", self.worker_python, str(Path(__file__)), "exec",
                 str(spec_file), str(out_file)],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.worker_timeout,
                env={**os.environ, "PYTHONPATH": str(PROJECT)},
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{entrypoint}: the worker produced no terminal within "
                f"{self.worker_timeout:g} s ({len(json.dumps(payload))} B payload)"
            ) from exc
        wall = time.perf_counter() - started
        if not out_file.is_file():
            raise RuntimeError(
                f"{entrypoint}: the worker process produced no outcome "
                f"(exit {proc.returncode})\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
            )
        outcome = json.loads(out_file.read_text())
        outcome["waited_seconds"] = waited
        outcome["wall_seconds"] = round(wall, 2)
        outcome["worker_exit"] = proc.returncode
        outcome["entrypoint"] = entrypoint
        self.runs.append(outcome)
        if self.verbose:
            timings = outcome.get("timings", {})
            metrics = outcome.get("metrics", {})
            print(
                f"   [slice] {entrypoint}: {outcome.get('status')} in {wall:.1f}s "
                f"(ready {timings.get('register_to_ready_ms', 0) / 1000:.1f}s, "
                f"attempt {timings.get('request_to_result_ms', 0) / 1000:.1f}s, "
                f"peak vram {float(metrics.get('peak_vram_bytes', 0)) / (1 << 20):.0f} MiB)",
                flush=True,
            )
        if outcome.get("status") != "TERMINAL_STATUS_SUCCEEDED":
            shortfall = [
                f["detail"] for f in outcome.get("faults", [])
                if f.get("reason") == "device_shortfall"
            ]
            if shortfall:
                raise _Shortfall(str(shortfall[0]).split(" - ")[0])
            cause = outcome.get("cause") or {}
            raise RuntimeError(
                f"{entrypoint}: the attempt did not succeed — {outcome.get('status')} "
                f"{cause.get('code', '')} {cause.get('origin', '')}: "
                f"{cause.get('detail') or outcome.get('safe_message') or ''}"
                f"{'' if cause else ' ' + str(proc.stderr)[-600:]}"
            )
        result = outcome.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"{entrypoint}: the terminal carried no typed result")
        # A per-call error is an ANSWER about that call — except when it is the card saying
        # no. An allocator OOM under a co-tenant is a resource condition at the other end of
        # the same fence a fill-time shortfall is, and it is retried the same way. Only when
        # EVERY call in the batch failed that way: one call OOMing while its neighbours
        # answered is a real per-call fact about that call's size.
        replies = result.get("replies") or []
        errors = [str(r.get("error", "")) for r in replies]
        if errors and all("out of memory" in e.lower() for e in errors):
            raise _Shortfall(f"every call OOMed: {errors[0][:120]}")
        return result


#: The card's idle floor on this box (the X server's own framebuffer). A slice is not over
#: when its process exits: the driver reclaims a dead context asynchronously, and the NEXT
#: slice's fill then meets a card that is still holding 4.8 GiB. Measured, not guessed —
#: two runs OOMed at fill with 24 MiB free while `nvidia-smi` showed no compute process.
IDLE_MIB = 400
SETTLE_SECONDS = 600.0


def _settle(need_mib: int = IDLE_MIB, timeout: float = SETTLE_SECONDS) -> float:
    """Wait, bounded, until the card has room for `need_mib`. Returns the seconds waited.

    Two things make this necessary, both measured. A slice is not over when its process
    exits — the driver reclaims a dead CUDA context asynchronously, and the next fill met a
    card still holding 4.8 GiB. And this box's card is SHARED with other agents, so the
    honest thing to do when another resident is live is to wait for room rather than to OOM
    a 4.9 GiB fill and call that a result.
    """
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        free = _gpu_free_mib()
        if free < 0 or free >= need_mib:
            break
        time.sleep(1.0)
    return round(time.perf_counter() - started, 2)


def _gpu_free_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return -1


# --------------------------------------------------------------------------- the worker


def run_one(spec: dict[str, Any], out: Path) -> int:
    """ONE slice, in THIS process. The whole of `exec` mode."""
    from cozy_runtime.internal.config import read_config
    from cozy_runtime.internal.local import LocalRequest, run_slice
    from cozy_runtime.internal.worker.session import WorkerOptions

    workspace = Path(spec["workspace"])
    home = workspace / "home"
    home.mkdir(parents=True, exist_ok=True)
    config = read_config(
        {
            "PATH": os.environ["PATH"],
            "HOME": os.environ["HOME"],
            "COZY_HOME": str(home),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        }
    )
    request = LocalRequest(
        entrypoint=spec["entrypoint"],
        payload=spec["payload"],
        outputs=(),  # a judge returns numbers and text: no output asset is ever granted
        request_id=spec["request_id"],
        deadline_ms=int(spec["deadline_ms"]),
    )
    outcome = run_slice(
        config,
        request,
        [spec["binding"]],
        workspace=workspace,
        release_id=spec["release_id"],
        options=WorkerOptions(root=workspace / "worker"),
    )
    document = {
        "status": outcome.status,
        "ended": outcome.ended,
        "exit_code": outcome.exit_code,
        "result": outcome.result,
        "result_schema_digest": outcome.result_schema_digest,
        "accepted": outcome.accepted,
        "ledger": outcome.ledger,
        "reconciliation": outcome.reconciliation,
        "timings": outcome.timings,
        "progress": outcome.progress,
        "faults": outcome.faults,
        "journal_deleted": outcome.journal_deleted,
        # `TerminalCause` — the typed reason a refusal happened, which is the whole point
        # of a refusal being typed. The transport prints it, so a red arm reads as its own
        # code rather than as "the attempt did not succeed".
        "cause": _cause(outcome.terminal or {}),
        "safe_message": (outcome.terminal or {}).get("safe_message", ""),
        "metrics": (outcome.terminal or {}).get("metrics") or {},
    }
    out.write_text(json.dumps(document))
    return 0


def _cause(terminal: dict[str, Any]) -> dict[str, Any]:
    """The typed refusal, with its enums spelled. `code: 1` is not a reason a person reads."""
    from cozy_runtime.protocol import worker_pb2 as pb

    cause = dict(terminal.get("cause") or {})
    if not cause:
        return {}
    with contextlib.suppress(ValueError):
        cause["code"] = pb.CauseCode.Name(int(cause.get("code", 0)))
    with contextlib.suppress(ValueError):
        cause["origin"] = pb.CauseOrigin.Name(int(cause.get("origin", 0)))
    return cause


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[1] == "exec":
        return run_one(json.loads(Path(argv[2]).read_text()), Path(argv[3]))
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
