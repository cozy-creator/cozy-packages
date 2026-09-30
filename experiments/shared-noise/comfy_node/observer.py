"""Bounded private quality evidence; no kernel, precision, RNG or weight changes."""

from __future__ import annotations
import contextvars
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
from contextlib import AbstractContextManager
import torch

CURRENT: contextvars.ContextVar[Record | None] = contextvars.ContextVar(
    "private_shared_noise_observer", default=None
)
OUTPUT = Path(
    "/home/fidika/.cozy/outputs/comfy-cozy-memory-20260929/analysis-shared-initial-noise/observations"
)
# At most two branches,512tokens,2048channels,F16/BF16. F32latents are <=1MiB.
MAX_TENSOR_BYTES = 2 * 512 * 2048 * 2
MAX_COPIED_BYTES = 16 * MAX_TENSOR_BYTES
MAX_EVENTS = 24


def typed(value: Any, depth: int = 0) -> Any:
    if depth > 24:
        raise ValueError("configuration nesting exceeds private bound")
    if isinstance(value, torch.dtype):
        if value not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ):
            raise ValueError("unsupported configuration Torch dtype")
        return {"type": "torch_dtype", "value": str(value)}
    if value is None:
        return {"type": "null"}
    if type(value) in (bool, int, str):
        return {"type": type(value).__name__, "value": value}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "hex": value.hex()}
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "items": [typed(x, depth + 1) for x in value]}
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {"type": "dict", "items": {k: typed(v, depth + 1) for k, v in value.items()}}
    raise ValueError("unsupported configuration value")


def configuration(value: Any) -> dict[str, Any]:
    tree = typed(dict(value))
    encoded = json.dumps(tree, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > 65536:
        raise ValueError("configuration exceeds private bound")
    return {"typed": tree, "sha256": hashlib.sha256(encoded).hexdigest()}


def allocator(device: Any) -> dict[str, int]:
    if torch.device(device).type != "cuda":
        return {}
    if not torch.cuda.is_initialized():
        raise RuntimeError("observer cannot initialize CUDA")
    stats = torch.cuda.memory_stats(device)
    return {
        k: int(stats[k])
        for k in (
            "allocated_bytes.all.current",
            "reserved_bytes.all.current",
            "num_ooms",
            "num_alloc_retries",
        )
    }


class Record:
    def __init__(self, engine: str, model: str, seed: int):
        if (model, seed) not in [("sdxl", 1005), ("anima", 1006)]:
            raise ValueError("unsupported private observer request")
        self.engine, self.model, self.seed = engine, model, seed
        self.events: list[dict[str, Any]] = []
        self.copied = 0
        self.copy_index = 0
        self.active = False
        self.group_calls = 0
        self.network_calls = 0
        self.branches: dict[str, Any] = {}
        self.generator: Any = None
        self.request: dict[str, Any] = {}
        self.started = time.time_ns()
        self.native_tid = threading.get_native_id()
        self.settings = {
            "torch": str(torch.__version__),
            "cuda_build": torch.version.cuda,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

    def tensor(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, torch.Tensor) or value.is_meta or value.layout != torch.strided:
            raise ValueError("unsupported observed tensor")
        size = value.numel() * value.element_size()
        if size > MAX_TENSOR_BYTES or self.copied + size > MAX_COPIED_BYTES:
            raise ValueError("private observation byte bound exceeded")
        self.copied += size
        self.copy_index += 1
        before = allocator(value.device)
        # Packing is CPU-only; at most two bounded CPU buffers, noGPUclone.
        cpu = value.detach().to(device="cpu", copy=True).contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy()
        checksum = hashlib.sha256(memoryview(raw)).hexdigest()
        row = {
            "copy_index": self.copy_index,
            "sha256": checksum,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "logical_bytes": size,
            "device": str(value.device),
            "allocator_before": before,
            "allocator_after": allocator(value.device),
        }
        if value.numel() <= 64:
            row["small_values"] = cpu.tolist()
        del raw, cpu
        return row

    def rng(self, device: Any) -> dict[str, Any]:
        answer = {
            "cpu": self.tensor(torch.get_rng_state()),
            "request_generator": self.tensor(self.generator.get_state())
            if self.generator is not None
            else None,
        }
        if torch.device(device).type == "cuda":
            if not torch.cuda.is_initialized():
                raise RuntimeError("uninitialized diagnostic device")
            answer["device"] = self.tensor(torch.cuda.get_rng_state(device))
        else:
            answer["device"] = None
        return answer

    def add(self, event: str, **values: Any) -> dict[str, Any]:
        if len(self.events) >= MAX_EVENTS:
            raise ValueError("private event bound exceeded")
        row = {"event": event, "native_tid": threading.get_native_id(), **values}
        self.events.append(row)
        return row

    def finish(self, error: BaseException | None) -> None:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        doc = {
            "engine": self.engine,
            "model": self.model,
            "seed": self.seed,
            "native_tid": self.native_tid,
            "events": self.events,
            "settings": self.settings,
            "request": self.request,
            "complete": error is None,
            "exception": None if error is None else type(error).__name__,
            "copied_bytes": self.copied,
            "max_tensor_bytes": MAX_TENSOR_BYTES,
            "max_copied_bytes": MAX_COPIED_BYTES,
            "scored": False,
            "scope": "firststep network prediction, not entire image equivalence",
        }
        (OUTPUT / f"{self.started}-{os.getpid()}-{self.engine}-{self.model}.json").write_text(
            json.dumps(doc, indent=2) + "\n"
        )


def observe_request(engine: str, model: str) -> Any:
    def decorate(fn: Any) -> Any:
        @functools.wraps(fn)
        def call(ctx: Any, payload: Any, *args: Any, **kwargs: Any) -> Any:
            record = Record(engine, model, int(payload.seed))
            record.request = {key: getattr(payload, key) for key in ("seed", "steps", "guidance")}
            for key in ("prompt", "negative_prompt"):
                value = getattr(payload, key)
                if not isinstance(value, str):
                    raise ValueError("unsupported prompt")
                record.request[key + "_sha256"] = hashlib.sha256(value.encode()).hexdigest()

            token = CURRENT.set(record)
            error = None
            try:
                return fn(ctx, payload, *args, **kwargs)
            except BaseException as exc:
                error = exc
                raise
            finally:
                CURRENT.reset(token)
                try:
                    record.finish(error)
                except Exception:
                    if error is None:
                        raise

        return call

    return decorate


def provider_return(model: str, seed: int, source_hash: str, source_shape: Any, value: Any) -> None:
    rec = CURRENT.get()
    if rec is not None:
        rec.add(
            "provider_return",
            source_sha256=source_hash,
            source_shape=source_shape,
            canonical_dtype="float32",
            returned=rec.tensor(value),
            rng_effect="original_draw_bypassed",
        )


def scheduler(rec: Record, owner: Any) -> dict[str, Any]:
    return {
        "class": type(owner).__module__ + "." + type(owner).__qualname__,
        "config": configuration(owner.config),
        "timesteps": rec.tensor(owner.timesteps),
        "sigmas": rec.tensor(owner.sigmas),
    }


def sdxl_first_state(
    schedule: Any, latents: Any, prompt: Any, pooled: Any, time_ids: Any, generator: Any
) -> None:
    rec = CURRENT.get()
    if rec is not None:
        rec.generator = generator
        rec.add(
            "initial_sampler_state",
            rng=rec.rng(latents.device),
            state=rec.tensor(latents),
            schedule=scheduler(rec, schedule),
            context=rec.tensor(prompt),
            pooled=rec.tensor(pooled),
            time_ids=rec.tensor(time_ids),
            branches=["negative", "positive"],
        )


def branch_digests(rec: Record, branches: list[str], tensors: dict[str, Any]) -> dict[str, Any]:
    if (
        not branches
        or len(set(branches)) != len(branches)
        or any(x not in ("positive", "negative") for x in branches)
    ):
        raise ValueError("unknown or repeated network branch mapping")
    result = {}
    for index, label in enumerate(branches):
        values = {}
        for name, tensor in tensors.items():
            if tensor.ndim and tensor.shape[0] == len(branches):
                values[name] = rec.tensor(tensor[index : index + 1])
            elif tensor.ndim == 0 or tensor.shape[0] == 1:
                values[name] = rec.tensor(tensor)
            else:
                raise ValueError("unsupported conditional batching geometry")
        result[label] = values
    return result


def observe_network(original: Any, model: str) -> Any:
    signature = inspect.signature(original)

    @functools.wraps(original)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        rec = CURRENT.get()
        if rec is None or not rec.active:
            return original(self, *args, **kwargs)
        rec.network_calls += 1
        if rec.network_calls > 2:
            raise ValueError("unexpected firststep network calls")
        bound = signature.bind(self, *args, **kwargs).arguments
        tensors = {k: v for k, v in bound.items() if isinstance(v, torch.Tensor)}
        values = {k: rec.tensor(v) for k, v in tensors.items()}
        extra = bound.get("added_cond_kwargs")
        if extra is not None:
            tensors.update({k: v for k, v in extra.items() if isinstance(v, torch.Tensor)})
            values.update(
                {k: rec.tensor(v) for k, v in extra.items() if isinstance(v, torch.Tensor)}
            )
        branch = (
            ["negative", "positive"]
            if model == "sdxl"
            else [
                k
                for k, v in rec.branches.items()
                if v["sha256"] == values.get("encoder_hidden_states", {}).get("sha256")
                and v["dtype"] == values.get("encoder_hidden_states", {}).get("dtype")
            ]
        )
        sample = tensors.get("sample", tensors.get("hidden_states"))
        if (
            sample is None
            or sample.shape[0] != len(branch)
            or (model == "anima" and len(branch) != 1)
        ):
            raise ValueError("ambiguous conditional branch mapping")
        row = rec.add(
            "network",
            ordinal=rec.network_calls,
            complete=False,
            inputs=values,
            branches=branch,
            output_kind="epsilon" if model == "sdxl" else "flow_velocity",
            model=model,
            config=configuration(self.config),
            training=bool(self.training),
            branch_inputs=branch_digests(rec, branch, tensors),
        )
        try:
            result = original(self, *args, **kwargs)
            value = (
                result.sample
                if hasattr(result, "sample")
                else result[0]
                if isinstance(result, tuple)
                else result
            )
            row.update(
                output=rec.tensor(value),
                branch_outputs=branch_digests(rec, branch, {"prediction": value}),
                complete=True,
            )
            return result
        except BaseException as exc:
            row["exception"] = type(exc).__name__
            raise

    return call


def first_group(enabled: bool) -> AbstractContextManager[None]:
    class Scope:
        def __enter__(self) -> None:
            self.rec = CURRENT.get()
            self.prior: bool | None = None
            if self.rec is not None and enabled:
                self.rec.group_calls += 1
                if self.rec.group_calls != 1:
                    raise ValueError("repeated firststep/retry requires separate evidence")
                self.prior = self.rec.active
                self.rec.active = True

        def __exit__(self, *exc: Any) -> None:
            if self.prior is not None and self.rec is not None:
                self.rec.active = self.prior

    return Scope()


def install_anima_observer(blocks: Any) -> None:
    block = blocks.sub_blocks["denoise.denoise"].sub_blocks["denoiser"]
    if getattr(type(block), "_shared_noise_observed", False):
        return
    upstream = type(block)

    def call(self: Any, components: Any, state: Any, i: int, t: Any) -> Any:
        rec = CURRENT.get()
        if rec is None or i != 0:
            return upstream.__call__(self, components, state, i, t)
        rec.branches = {
            name: rec.tensor(getattr(state, field))
            for name, field in [
                ("positive", "prompt_embeds"),
                ("negative", "negative_prompt_embeds"),
            ]
        }
        rec.add(
            "initial_sampler_state",
            rng=rec.rng(state.latents.device),
            state=rec.tensor(state.latents),
            schedule=scheduler(rec, components.scheduler),
            context=rec.branches,
            timestep=rec.tensor(state.timestep),
        )
        with first_group(True):
            return upstream.__call__(self, components, state, i, t)

    block.__class__ = type(
        "Observed" + upstream.__name__,
        (upstream,),
        {"__call__": call, "__module__": __name__, "_shared_noise_observed": True},
    )


def observe_generator(generator: Any) -> None:
    record = CURRENT.get()
    if record is not None:
        record.generator = generator
