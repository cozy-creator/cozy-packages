"""Private, default-disabled full-request decoder probe; never a scored run."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from typing import Annotated, Any, cast

import torch
import cozy_runtime.author as cozy_author
from cozy_runtime.author import (
    App,
    AssetBound,
    Context,
    FileAsset,
    ImageFrame,
    Loader,
    Outputs,
    Telemetry,
    UnsupportedInput,
    uses_components,
)
from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
from diffusers.models.autoencoders.vae import DecoderOutput
from safetensors.torch import load as load_tensors, save as save_tensors
from transformers import Qwen3Config, Qwen3Model
from transformers import initialization as transformer_init

from . import (
    AnimaModel,
    AnimaPipeline,
    GenerateInput,
    ImageOutput,
    _AnimaVae,
    _BUCKETS,
    _Phases,
    _ROOT,
    _tokenizer,
    render_request,
    _SAVE,
)

app = App()
ARG_LIMIT = 1 << 20
RAW_LIMIT = 16 << 20
RGB_LIMIT = 4 << 20
JSON_LIMIT = 1 << 20
TRACE_LIMIT = 128 << 20
EXPORT_LIMITS = {
    "image": 64 << 20,
    "argument": ARG_LIMIT,
    "scalar_raw": RAW_LIMIT,
    "broadcast_raw": RAW_LIMIT,
    "scalar_rgb": RGB_LIMIT,
    "broadcast_rgb": RGB_LIMIT,
    "report": JSON_LIMIT,
    "trace": TRACE_LIMIT,
}
CACHE_NAMES = (
    "_conv_num",
    "_conv_idx",
    "_feat_map",
    "_enc_conv_num",
    "_enc_conv_idx",
    "_enc_feat_map",
)
DIAG_NAMES = (
    "_diag_mode",
    "_diag_profile",
    "_diag_capture",
    "_diag_argument",
    "_diag_baseline",
    "_diag_fast",
    "_diag_scalar",
)
FROZEN_PAYLOAD = {
    "prompt": "a peaceful alpine lake surrounded by pine forests and snow-covered mountains, clear blue sky, landscape, no people",
    "negative_prompt": "people, person, human, girl, boy, nude, nudity, naked, nsfw, low quality, worst quality, blurry",
    "steps": 30,
    "guidance": 4.5,
    "seed": 1006,
    "aspect_ratio": "1:1",
    "megapixels": 1,
    "quality_prefix": "",
    "cfg_interval_start": 0,
    "cfg_interval_stop": 1,
    "first_block_cache": 0.0,
}
FROZEN_CHECKPOINT = "sha256:5d6a05188427ba195e150f5a1225e965bc02bef3a5f8c8f372e81579d22f5c38"
BASELINE_RGB = "e5783599c8d7e048830fcd672e1831eb2cb87a0ae3d22132ab19c04b5acee728"


def _require_frozen_probe(payload: Any, checkpoint: str) -> None:
    if any(
        getattr(getattr(payload, name), "value", getattr(payload, name)) != value
        for name, value in FROZEN_PAYLOAD.items()
    ):
        raise RuntimeError("probe fields differ from the unchanged frozen full request")
    if checkpoint != FROZEN_CHECKPOINT:
        raise RuntimeError("probe checkpoint differs from the frozen model snapshot")


class DiagnosticInput(GenerateInput):
    probe: bool = False


class DiagnosticOutput(ImageOutput):
    argument: Annotated[
        FileAsset | None, AssetBound(max_bytes=1 << 20, media_types=("application/octet-stream",))
    ] = None
    scalar_raw: Annotated[
        FileAsset | None, AssetBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
    ] = None
    broadcast_raw: Annotated[
        FileAsset | None, AssetBound(max_bytes=16 << 20, media_types=("application/octet-stream",))
    ] = None
    scalar_rgb: Annotated[
        FileAsset | None, AssetBound(max_bytes=4 << 20, media_types=("application/octet-stream",))
    ] = None
    broadcast_rgb: Annotated[
        FileAsset | None, AssetBound(max_bytes=4 << 20, media_types=("application/octet-stream",))
    ] = None
    report: Annotated[
        FileAsset | None, AssetBound(max_bytes=1 << 20, media_types=("application/json",))
    ] = None
    trace: Annotated[
        FileAsset | None, AssetBound(max_bytes=128 << 20, media_types=("application/json",))
    ] = None


def _require_native_cuda(tensor: Any, role: str) -> None:
    if (
        tensor.device.type != "cuda"
        or not tensor.is_contiguous()
        or tensor.is_conj()
        or tensor.is_neg()
    ):
        raise RuntimeError(f"{role}: unsafe CUDA layout; no implicit device packing")
    shape = (1, 16, 1, 128, 128) if role == "argument" else (1, 3, 1, 1024, 1024)
    if tuple(tensor.shape) != shape or tensor.dtype not in (torch.bfloat16, torch.float32):
        raise RuntimeError(f"{role}: unexpected full-request tensor shape/dtype")
    if role == "argument" and tensor.dtype != torch.bfloat16:
        raise RuntimeError("argument: expected actual normalized BF16 decode argument")


def _host_tensor_bytes(
    tensor: Any, role: str, native_layout: dict[str, str] | None = None
) -> bytes:
    if tensor.device.type != "cpu" or tensor.is_conj() or tensor.is_neg():
        raise RuntimeError("serialization requires materialized CPU tensor")
    data = save_tensors(
        {"tensor": tensor.contiguous()}, metadata={"boundary": role, **(native_layout or {})}
    )
    if len(data) > (ARG_LIMIT if role == "argument" else RAW_LIMIT):
        raise RuntimeError(f"{role}: full tensor artifact exceeds its declared bound")
    return data


def _drain_native(tensor: Any, role: str) -> bytes:
    _require_native_cuda(tensor, role)
    layout = {
        "native_shape": json.dumps(list(tensor.shape)),
        "native_stride": json.dumps(list(tensor.stride())),
        "native_dtype": str(tensor.dtype),
        "native_device": str(tensor.device),
    }
    cpu = tensor.to(device="cpu", dtype=tensor.dtype, non_blocking=False)
    return _host_tensor_bytes(cpu.contiguous(), role, layout)


def _rgb(data: bytes) -> bytes:
    tensor = load_tensors(data)["tensor"]
    image = (tensor[:, :, 0] * 0.5 + 0.5).clamp(0, 1)[0]
    pixels = (image.clamp(0, 1) * 255).to(dtype=torch.uint8).permute(1, 2, 0).contiguous()
    return pixels.numpy().tobytes()


def _exact_raw(a: bytes, b: bytes) -> bool:
    first, second = load_tensors(a)["tensor"], load_tensors(b)["tensor"]
    return bool(
        first.dtype == second.dtype
        and first.shape == second.shape
        and torch.equal(first.contiguous().view(torch.uint8), second.contiguous().view(torch.uint8))
    )


def _raw_bit_digest(data: bytes) -> str:
    tensor = load_tensors(data)["tensor"]
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _profiled_blend(self: Any, a: Any, b: Any, extent: int, axis: int) -> bool:
    # Only profiler scopes differ from the literal reviewed helper. CPU source
    # projection strips these scopes and proves its body against sourcePR373.
    if (
        type(extent) is not int
        or extent <= 0
        or extent & (extent - 1)
        or a.dtype != b.dtype
        or a.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64)
        or extent > 2 / torch.finfo(a.dtype).eps
        or a.device != b.device
        or a.device.type not in ("cpu", "cuda")
        or a.ndim != 5
        or b.ndim != 5
        or a.layout != torch.strided
        or b.layout != torch.strided
        or any(a.shape[i] != b.shape[i] for i in range(5) if i != axis)
        or not a.is_contiguous()
        or not b.is_contiguous()
        or not a.numel()
        or not b.numel()
        or (torch.is_grad_enabled() and (a.requires_grad or b.requires_grad))
    ):
        return False
    a_storage, b_storage = a.untyped_storage(), b.untyped_storage()
    a_start, b_start = a_storage.data_ptr(), b_storage.data_ptr()
    if a_start < b_start + b_storage.nbytes() and b_start < a_start + a_storage.nbytes():
        return False
    shape = [1] * 5
    shape[axis] = extent
    source = [slice(None)] * 5
    target = [slice(None)] * 5
    source[axis] = slice(-extent, None)
    target[axis] = slice(extent)
    a_overlap, b_overlap = a[tuple(source)], b[tuple(target)]
    with torch.profiler.record_function("diagnostic.blend.finite_reductions_and_item"):
        if not (torch.isfinite(a_overlap).all() & torch.isfinite(b_overlap).all()).item():
            return False
    with torch.profiler.record_function("diagnostic.blend.coefficient_construct_copy"):
        coefficients = torch.tensor(
            [(1 - i / extent, i / extent) for i in range(extent)],
            dtype=b.dtype,
            device=b.device,
        )
    with torch.profiler.record_function("diagnostic.blend.left_product"):
        left = a_overlap * coefficients[:, 0].reshape(shape)
    with torch.profiler.record_function("diagnostic.blend.right_product"):
        right = b_overlap * coefficients[:, 1].reshape(shape)
    with torch.profiler.record_function("diagnostic.blend.add_and_copy"):
        b[tuple(target)] = left + right
    return True


def _initialize_diagnostic(vae: Any) -> None:
    vae._diag_mode = "scalar"
    vae._diag_profile = False
    vae._diag_capture = False
    vae._diag_argument = b""
    vae._diag_baseline = b""
    vae._diag_fast = vae._diag_scalar = 0


def _declare_diagnostic_retry_state(vae: Any) -> None:
    names = CACHE_NAMES + DIAG_NAMES

    def save() -> dict[str, tuple[Any, list[Any] | None]]:
        return {
            name: (value, list(value) if isinstance(value, list) else None)
            for name in names
            if name in vars(vae)
            for value in (vars(vae)[name],)
        }

    def restore(state: dict[str, tuple[Any, list[Any] | None]]) -> None:
        for name in names:
            vars(vae).pop(name, None)
        for name, (value, items) in state.items():
            if items is not None:
                value[:] = items
            vars(vae)[name] = value

    cozy_author.retry_state(vae, save, restore)


class _DiagnosticVae(_AnimaVae):
    _diag_mode: str
    _diag_profile: bool
    _diag_capture: bool
    _diag_argument: bytes
    _diag_baseline: bytes
    _diag_fast: int
    _diag_scalar: int

    def _blend(self, a: Any, b: Any, extent: int, axis: int) -> bool:
        if self._diag_mode == "scalar":
            self._diag_scalar += 1
            return False
        fast = bool(
            _profiled_blend(self, a, b, extent, axis)
            if self._diag_profile
            else super()._blend(a, b, extent, axis)
        )
        if fast:
            self._diag_fast += 1
        else:
            self._diag_scalar += 1
        return fast

    def blend_v(self, a: Any, b: Any, blend_extent: int) -> Any:
        with (
            torch.profiler.record_function("diagnostic.blend.vertical." + self._diag_mode)
            if self._diag_profile
            else nullcontext()
        ):
            return super().blend_v(a, b, blend_extent)

    def blend_h(self, a: Any, b: Any, blend_extent: int) -> Any:
        with (
            torch.profiler.record_function("diagnostic.blend.horizontal." + self._diag_mode)
            if self._diag_profile
            else nullcontext()
        ):
            return super().blend_h(a, b, blend_extent)

    def decode(
        self,
        z: Any,
        return_dict: bool = True,
        *,
        blend_mode: str = "scalar",
        profile_decode: bool = False,
    ) -> Any:
        if blend_mode not in ("scalar", "broadcast"):
            raise RuntimeError("unknown frozen decoder blend mode")
        previous = self._diag_mode
        previous_profile = self._diag_profile
        self._diag_mode = blend_mode
        self._diag_profile = profile_decode
        self._diag_fast = self._diag_scalar = 0
        try:
            if self._diag_capture:
                _require_native_cuda(z, "argument")
            with (
                torch.profiler.record_function("diagnostic.decode.tiling_and_concatenation")
                if profile_decode
                else nullcontext()
            ):
                decoded = cast(Any, super()).decode(z, return_dict=False)[0]
            if self._diag_capture:
                self._diag_argument = _drain_native(z, "argument")
                self._diag_baseline = _drain_native(decoded, "decoded")
            return DecoderOutput(sample=decoded) if return_dict else (decoded,)
        finally:
            self._diag_mode = previous
            self._diag_profile = previous_profile


def _save_bounded(out: Outputs, data: bytes, name: str) -> FileAsset:
    if len(data) > EXPORT_LIMITS[name]:
        raise RuntimeError(f"{name}: incomplete proof artifact exceeds declared bound")
    return out.save_bytes(
        data,
        media_type="application/json"
        if name in ("report", "trace")
        else "application/octet-stream",
    )


def _diagnostic_result(baseline: ImageOutput, **files: Any) -> DiagnosticOutput:
    return DiagnosticOutput(
        baseline.image,
        baseline.width,
        baseline.height,
        baseline.steps,
        baseline.guidance,
        baseline.digest,
        **files,
    )


class DiagnosticModel(AnimaModel):
    pipe: DiagnosticPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(DiagnosticPipeline, factory=_build_diagnostic_pipeline)

    @uses_components("vae")
    def configure_probe(self, enabled: bool) -> None:
        vae = self.pipe.components["vae"]
        _initialize_diagnostic(vae)
        vae._diag_capture = enabled

    @uses_components("vae")
    def probe_decode(self, argument: bytes, blend_mode: str) -> tuple[bytes, bytes, dict[str, Any]]:
        vae = self.pipe.components["vae"]
        cpu = load_tensors(argument)["tensor"]
        z = cpu.to(device=next(vae.parameters()).device, dtype=cpu.dtype)
        _require_native_cuda(z, "argument")
        torch.cuda.synchronize()
        event_start = torch.cuda.Event(enable_timing=True)
        event_end = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        event_start.record()
        with torch.profiler.record_function("diagnostic.decode." + blend_mode):
            decoded = vae.decode(z, return_dict=False, blend_mode=blend_mode, profile_decode=True)[
                0
            ]
        event_end.record()
        event_end.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000
        with torch.profiler.record_function("diagnostic.host_drain." + blend_mode):
            raw = _drain_native(decoded, "decoded")
            if _drain_native(z, "argument") != argument:
                raise RuntimeError("decoder argument bytes changed")
        with torch.profiler.record_function("diagnostic.cpu_postprocess_rgb." + blend_mode):
            rgb = _rgb(raw)
        return (
            raw,
            rgb,
            {
                "mode": blend_mode,
                "decode_host_wall_ms": wall_ms,
                "decode_cuda_event_ms": event_start.elapsed_time(event_end),
                "raw_bits_sha256": _raw_bit_digest(raw),
                "raw_artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
                "fast_blends": vae._diag_fast,
                "scalar_blends": vae._diag_scalar,
            },
        )


@app.entrypoint
def generate(
    ctx: Context, payload: DiagnosticInput, model: DiagnosticModel, out: Outputs, tel: Telemetry
) -> DiagnosticOutput:
    if payload.probe:
        _require_frozen_probe(payload, model.checkpoint_ref)
    model.configure_probe(payload.probe)
    vae = model.pipe.components["vae"]
    try:
        baseline = _generate_baseline(ctx, payload, model, out, tel)
        if not payload.probe:
            return _diagnostic_result(baseline)
        if baseline.digest != BASELINE_RGB:
            raise RuntimeError("ordinary scalar baseline differs from qualified seed1006 RGB")
        argument, captured = vae._diag_argument, vae._diag_baseline
        vae._diag_capture = False
        if not argument or not captured:
            raise RuntimeError("missing full generation decoder capture")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        ) as prof:
            old_raw, old_rgb, old_record = model.probe_decode(argument, "scalar")
            new_raw, new_rgb, new_record = model.probe_decode(argument, "broadcast")
        operators = [
            {
                "name": row.key,
                "calls": row.count,
                "cpu_total_us": row.cpu_time_total,
                "device_total_us": row.device_time_total,
            }
            for row in prof.key_averages()
        ]
        if not any(row["device_total_us"] > 0 for row in operators):
            raise RuntimeError("incomplete CUDA profile; no CPU-only substitution")
        if not _exact_raw(captured, old_raw) or not _exact_raw(old_raw, new_raw):
            raise RuntimeError("raw decoder bits differ; probe is not qualified")
        if old_rgb != new_rgb or hashlib.sha256(old_rgb).hexdigest() != baseline.digest:
            raise RuntimeError("RGB/baseline boundary differs; probe is not qualified")
        trace_path = out.temporary_file(".json")
        prof.export_chrome_trace(str(trace_path))
        trace = trace_path.read_bytes()
        report = json.dumps(
            {
                "producer_complete": True,
                "runtime_gate_pending": True,
                "scored": False,
                "source_profile_decode_calls": 2,
                "profile_regions": 1,
                "tile": 256,
                "stride": 192,
                "argument_sha256": hashlib.sha256(argument).hexdigest(),
                "argument_bits_sha256": _raw_bit_digest(argument),
                "baseline_raw_internal_sha256": hashlib.sha256(captured).hexdigest(),
                "baseline_raw_bits_sha256": _raw_bit_digest(captured),
                "baseline_rgb_sha256": baseline.digest,
                "checkpoint_ref": model.checkpoint_ref,
                "request": FROZEN_PAYLOAD,
                "raw_bits_exact": True,
                "rgb_exact": True,
                "modes": [old_record, new_record],
                "operators": operators,
                "record_shapes": False,
                "profile_memory": False,
                "with_stack": False,
                "trace_is_file_cap_not_profiler_RAM_cap": True,
                "cuda_event_boundary": "same current stream; completion before host drain",
                "cpu_postprocess_requires_baseline_RGB_match": True,
                "concatenation_range_includes_entire_tiler": True,
                "runtime_invocation_retry_residency_evidence": "external accepted Worker receipt and closed ledger required",
            }
        ).encode()
        blobs = {
            "argument": argument,
            "scalar_raw": old_raw,
            "broadcast_raw": new_raw,
            "scalar_rgb": old_rgb,
            "broadcast_rgb": new_rgb,
            "report": report,
            "trace": trace,
        }
        if baseline.image.size_bytes + sum(map(len, blobs.values())) > sum(EXPORT_LIMITS.values()):
            raise RuntimeError("aggregate diagnostic inventory exceeds frozen bound")
        files = {name: _save_bounded(out, data, name) for name, data in blobs.items()}
        return _diagnostic_result(baseline, **files)
    finally:
        _initialize_diagnostic(vae)


class DiagnosticPipeline(AnimaPipeline):
    def __init__(self, config: Any) -> None:
        mapping = config.mapping()
        with transformer_init.no_init_weights():
            transformer = CosmosTransformer3DModel.from_config(mapping["transformer"]).to(
                torch.bfloat16
            )
            text_encoder: Any = Qwen3Model(Qwen3Config(**mapping["text_encoder"]))
            text_encoder.to(dtype=torch.bfloat16)
            text_conditioner = AnimaTextConditioner.from_config(mapping["text_conditioner"]).to(
                torch.bfloat16
            )
            vae = _DiagnosticVae.from_config(mapping["vae"]).to(torch.bfloat16)

        for component in (transformer, text_encoder, text_conditioner, vae):
            component.eval()
        # Cosmos' uncached forward (including rotary embeddings) only reads module
        # state and arguments. First-block caching gets a separate request contract
        # for its stateful hooks and forward replacement below.
        cozy_author.pure(transformer)
        cozy_author.pure(text_conditioner)
        # The modular encoder passes no past_key_values: any KV cache is local
        # to this call. Dynamic RoPE rewrites buffers and has no purity contract.
        if getattr(text_encoder.rotary_emb, "rope_type", None) == "default":
            cozy_author.pure(text_encoder)
        _initialize_diagnostic(vae)
        _declare_diagnostic_retry_state(vae)
        vae.enable_tiling()
        self.scheduler_config = mapping["scheduler"]
        self.tokenizer = _tokenizer(_ROOT / "tokenizer")
        self.t5_tokenizer = _tokenizer(_ROOT / "t5_tokenizer")
        self.components: dict[str, Any] = {
            "transformer": transformer,
            "text_encoder": text_encoder,
            "text_conditioner": text_conditioner,
            "vae": vae,
        }


def _build_diagnostic_pipeline(config: Any) -> DiagnosticPipeline:
    return DiagnosticPipeline(config)


def _generate_baseline(
    ctx: Context,
    payload: GenerateInput,
    model: DiagnosticModel,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    """Generate one native-resolution Anima image."""
    width, height = _BUCKETS[(payload.aspect_ratio, payload.megapixels)]
    steps = payload.steps
    if payload.cfg_interval_start > payload.cfg_interval_stop:
        raise UnsupportedInput(
            "cfg_interval_start must not exceed cfg_interval_stop", code="cfg_interval"
        )
    with _Phases(tel) as phases:
        images = render_request(
            model,
            payload.quality_prefix + payload.prompt,
            payload.negative_prompt,
            width,
            height,
            steps,
            payload.guidance,
            (payload.cfg_interval_start, payload.cfg_interval_stop),
            payload.first_block_cache,
            payload.seed,
            phases,
        )
        # Still `decoding`: the pipeline left that bracket open and the host copy below is
        # the tail of the same work.
        image = images[0]
        pixels = (image.clamp(0, 1) * 255).to("cpu", dtype=torch.uint8)
        if pixels.ndim == 3 and pixels.shape[0] == 3:
            pixels = pixels.permute(1, 2, 0)
        pixels = pixels.contiguous()
        rgb = bytes(pixels.numpy().tobytes())
        phases.enter(_SAVE)
        asset = out.save_image(ImageFrame(width, height, rgb), format="webp")
    return ImageOutput(
        asset, width, height, steps, payload.guidance, hashlib.sha256(rgb).hexdigest()
    )
