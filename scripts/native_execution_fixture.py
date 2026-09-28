"""Real Runtime journal and TensorFS FD bindings for small package numerical proofs."""

from __future__ import annotations

import hashlib
import socket
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import msgspec
from cozy_runtime.author import Context, ModelArtifact, canonical_json
from cozy_runtime.author._executor_requests import (
    Answer,
    Reply,
    Request,
    WriterAdopt,
    WriterOutput,
    WriterSource,
    encode,
    respond,
)
from cozy_runtime.internal.seam import Channel
from cozy_runtime.internal.weights_writer import ExecutionStorage, WriterAttempt, WriterBinding
from cozy_runtime.internal.worker.grants import MODEL_PREFIX
from cozy_runtime.internal.worker.weights import WeightsExchange
from cozy_runtime.internal.worker.workspace import Workspace
from cozy_runtime.protocol import documents
from cozy_runtime.protocol import worker_pb2 as pb
from tensorfs.derived import OutputCapability


class NativeExecution:
    def __init__(
        self,
        store: Any,
        root: Path,
        name: str,
        models: dict[str, ModelArtifact],
        outputs: dict[str, int],
        *,
        epoch: int = 1,
        after_checkpoint: Callable[[Any], None] | None = None,
    ) -> None:
        self.spool = root / f"{name}-{epoch}"
        self.spool.mkdir(parents=True, exist_ok=True)
        destinations = [
            {"output_id": slot, "max_bytes": maximum, "mime_type": "application/x-cozytensors"}
            for slot, maximum in outputs.items()
        ]
        spec = {
            "format": "cozy.worker.v1.InvocationSpec/1",
            "deadline_unix_ms": 9_000_000_000_000,
            "installation_id": "local-" + "11" * 16,
            "payload_digest": "sha256:" + "12" * 32,
            "job": {
                "installation_id": "local-" + "13" * 16,
                "job_descriptor_id": "14" * 32,
                "publication_contract": {"grant_id": "15" * 32, "outputs": destinations},
            },
            "inputs": [
                {
                    "input_id": MODEL_PREFIX + parameter,
                    "digest": model.manifest.digest,
                    "length": model.manifest.length,
                }
                for parameter, model in models.items()
            ],
            "outputs": destinations,
        }
        raw = canonical_json.encode(spec)
        self.attempt = SimpleNamespace(
            request_id=name,
            attempt=epoch,
            digest=hashlib.sha256(raw).digest(),
            canceling="",
            state="running",
            spool=self.spool,
            spec=spec,
            weights_receipts={},
            weights_work_fingerprint="sha256:" + "16" * 32,
        )
        workspace = Workspace(Path(store.root))
        workspace.accept(
            "package-proof",
            pb.AttemptOffer(
                request_id=name,
                attempt_ordinal=epoch,
                invocation_spec_digest=self.attempt.digest,
                invocation_spec_canonical_bytes=raw,
            ),
        )
        self.owner = WeightsExchange(
            store_root=Path(store.root),
            workspace=workspace,
            stop=threading.Event(),
            owner_scope=lambda: "package-proof",
        )
        self.checkpointed = False

        def checkpoint(
            attempt: WriterAttempt,
            transaction: str,
            binding: WriterBinding,
            facts: Mapping[str, object],
        ) -> None:
            self.owner.record_checkpoint(attempt, transaction, binding, facts)
            self.checkpointed = True
            if after_checkpoint is not None:
                after_checkpoint(facts)

        self.owner.writer_broker.record_checkpoint = checkpoint
        self.client = ExecutionStorage(self.spool, self.exchange, outputs)
        self.replayed_outputs: set[str] = set()

    def exchange[A: Answer](self, request: Request, into: type[A], /) -> A:
        """One writer request across a real seam socket pair, its descriptor included."""

        def handle(sent: Request) -> Reply:
            assert isinstance(sent, WriterSource | WriterOutput | WriterAdopt)
            return self.owner.writer_broker.handle(self.attempt, sent)

        left, right = socket.socketpair()
        with left, right:
            executor, worker = Channel(left), Channel(right)
            executor.send({"event": "request", "seq": 1, **encode(request)})
            frame = worker.recv()
            assert frame is not None
            answer, handoff = respond(frame, handle)
            try:
                worker.send(answer)
                if handoff is not None:
                    worker.send_descriptor(handoff)
            finally:
                if handoff is not None:
                    handoff.close()
            reply = executor.recv()
            assert reply is not None and reply["seq"] == 1
            if reply.get("descriptor") is True:
                reply["descriptor"] = executor.recv_descriptor()
            return msgspec.convert(reply, into)

    def context(self) -> Context:
        def open_output(slot: str, definition: Any) -> Any:
            transaction = self.client.open_output(slot, definition)
            if transaction.receipt is not None:
                self.replayed_outputs.add(slot)
            return transaction

        return Context(
            self.attempt.request_id,
            time.monotonic() + 600,
            _tensorfs_source=lambda model: self.client.source(model.checkpoint_ref),
            _tensorfs_output=lambda slot: OutputCapability(
                lambda definition: open_output(slot, definition)
            ),
            _tensorfs_adopt=self.client.adopt_model,
        )

    def __enter__(self) -> NativeExecution:
        return self

    def __exit__(self, *exc: object) -> None:
        # The component harness ends this execution after inspecting native work;
        # it does not claim a Creator product outcome or release retained tensors.
        body, digest = documents.identity(
            pb.AttemptOutcomeBody(
                request_id=self.attempt.request_id,
                attempt_ordinal=self.attempt.attempt,
                invocation_spec_digest=documents.spell(self.attempt.digest),
                status=pb.OUTCOME_STATUS_ABANDONED,
                execution_started=True,
            )
        )
        assert self.owner.workspace is not None
        self.owner.workspace.outcome(
            "package-proof",
            pb.AttemptOutcome(
                request_id=self.attempt.request_id,
                attempt_ordinal=self.attempt.attempt,
                invocation_spec_digest=self.attempt.digest,
                outcome_id=f"fixture-{self.attempt.attempt}",
                outcome_digest=digest,
                outcome_canonical_bytes=body,
            ),
        )
        self.owner.close()
