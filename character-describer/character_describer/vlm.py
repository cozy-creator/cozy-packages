"""A Transformers image-text model built from a Cozy artifact, and the recipe that prepares one."""

from __future__ import annotations

import base64
import json
import zlib
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import msgspec
import numpy as np
import torch
import transformers
from cozy_runtime.author import (
    Config,
    Context,
    Loader,
    Model,
    ModelArtifact,
    Telemetry,
    UnsupportedInput,
    canonical_json,
    uses_components,
)
from cozy_runtime.derive.quantization import QuantizationSource
from PIL.Image import Image
from tensorfs.derived import Config as DerivedConfig
from tensorfs.derived import Derivation, Part, PartSource, Target, Tensor
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig, PreTrainedTokenizerFast
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import WeightConverter, rename_source_key
from transformers.initialization import no_init_weights

COMPONENT = "model"
DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32", torch.int64: "i64"}
_CAST_LIMIT = 64 << 20


class ArtifactConfig(msgspec.Struct, frozen=True):
    model_config: dict[str, Any]
    files: dict[str, str]


def _blob(raw: bytes) -> str:
    # Config refuses strings containing `://`, which tokenizer merges contain.
    return base64.urlsafe_b64encode(zlib.compress(raw, 6)).decode("ascii")


def _unblob(value: str) -> bytes:
    return zlib.decompress(base64.urlsafe_b64decode(value))


def _strip(value: Any) -> Any:
    """Drop upstream provenance such as `_name_or_path`: a construction config holds no carrier."""
    if isinstance(value, dict):
        return {
            key: _strip(item) for key, item in value.items()
            if not key.lower().endswith(("_path", "_dir", "_file", "_url", "_repo"))
        }
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _persist_buffers(module: torch.nn.Module) -> torch.nn.Module:
    """Runtime fills only stored bytes, so every buffer must be part of the checkpoint."""
    for sub in module.modules():
        sub._non_persistent_buffers_set.clear()
    return module


def _construct(model_config: Mapping[str, Any]) -> torch.nn.Module:
    """Every parameter is filled from the checkpoint; buffers keep their constructed values."""
    config = AutoConfig.for_model(**model_config)
    with no_init_weights():
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)
    model.tie_weights()
    return _persist_buffers(model.eval())


def _processor(members: Mapping[str, bytes]) -> Any:
    """The upstream processor, built from bytes: model loading never touches the filesystem."""
    image = json.loads(members["preprocessor_config.json"])
    processor_class = getattr(transformers, image.pop("processor_class"))
    image_processor = getattr(transformers, image.pop("image_processor_type"))(**image)
    extra = json.loads(members.get("processor_config.json", b"{}"))
    extra.pop("processor_class", None)
    if "video_preprocessor_config.json" in members:
        video = json.loads(members["video_preprocessor_config.json"])
        video.pop("processor_class", None)
        extra["video_processor"] = getattr(transformers, video.pop("video_processor_type"))(**video)
    tokenizer_config = json.loads(members["tokenizer_config.json"])
    for key in ("added_tokens_decoder", "tokenizer_file", "processor_class", "chat_template"):
        tokenizer_config.pop(key, None)
    named = tokenizer_config.pop("tokenizer_class", "")
    tokenizer_class = getattr(transformers, named, None) if named != "PreTrainedTokenizer" else None
    tokenizer = (tokenizer_class or PreTrainedTokenizerFast)(
        tokenizer_object=Tokenizer.from_str(members["tokenizer.json"].decode()), **tokenizer_config
    )
    if "chat_template.jinja" in members:
        chat = members["chat_template.jinja"].decode()
    else:
        chat = json.loads(members["chat_template.json"])["chat_template"]
    return processor_class(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat, **extra)


class Pipeline:
    def __init__(self, config: Config) -> None:
        artifact = msgspec.convert(config.mapping(), ArtifactConfig)
        members = {name: _unblob(value) for name, value in artifact.files.items()}
        model = _construct(artifact.model_config)
        if "generation_config.json" in members:
            model.generation_config = GenerationConfig(**json.loads(members["generation_config.json"]))
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model}
        self.processor = _processor(members)


class Describer(Model[Pipeline]):
    """One greedy reply about one image."""

    pipe: Pipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)

    def unload(self, loader: Loader) -> None:
        del self.pipe

    @uses_components(COMPONENT)
    def describe(self, image: Image, prompt: str, *, max_new_tokens: int) -> str:
        model = self.pipe.components[COMPONENT]
        processor = self.pipe.processor
        if type(processor).__name__ == "LlavaProcessor":
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        configured = model.generation_config.eos_token_id
        stops = [*(configured if isinstance(configured, list) else [configured]), processor.tokenizer.eos_token_id]
        stop = [token for token in dict.fromkeys(stops) if token is not None]
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=stop, pad_token_id=processor.tokenizer.pad_token_id or stop[0],
            )
        reply = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return " ".join(str(reply).split())


def prepare(source: QuantizationSource, ctx: Context, tel: Telemetry) -> ModelArtifact:
    """Rename, tie, cast and complete raw upstream weights exactly as Transformers loads them,
    adding the upstream configuration and processor files that ship in `upstream/`."""
    root = files(__package__).joinpath("upstream")
    members = {entry.name: entry.read_bytes() for entry in root.iterdir() if entry.is_file()}
    model_config = _strip(json.loads(members.pop("config.json")))
    with tel.stage("constructing the reference model"):
        model = _construct(model_config)
    expected = model.state_dict()
    capability = ctx.tensorfs_source(source)
    inspection = capability.inspect()
    if len(inspection.components) != 1:
        raise UnsupportedInput("the raw source must hold exactly one component")
    ((component, tensors),) = inspection.components.items()
    transforms = get_model_conversion_mapping(model)
    renamings = [t for t in transforms if not isinstance(t, WeightConverter)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    mapped: dict[str, str] = {}
    for key in tensors:
        target, converted = rename_source_key(key, renamings, converters, model.base_model_prefix, expected)
        if converted is not None:
            raise UnsupportedInput(f"{key} needs a Transformers weight conversion this recipe lacks")
        if target in expected:
            mapped[target] = key
    tied: dict[int, list[str]] = {}
    for key, parameter in model.named_parameters(remove_duplicate=False):
        tied.setdefault(id(parameter), []).append(key)
    for group in tied.values():
        origin = next((mapped[key] for key in group if key in mapped), None)
        if origin is not None:
            for key in group:
                mapped.setdefault(key, origin)
    buffers = dict(model.named_buffers())
    missing = [key for key in expected if key not in mapped and key not in buffers]
    if missing:
        raise UnsupportedInput(f"the source lacks {len(missing)} parameters, first {missing[0]}")
    encoding = next(iter(tensors.values())).encoding
    rows: dict[str, Tensor] = {}
    written: dict[str, bytes | str] = {}
    for key, value in expected.items():
        dtype, shape = DTYPES[value.dtype], tuple(value.shape)
        if key not in mapped:
            written[key] = value.contiguous().view(torch.uint8).numpy().tobytes()
            rows[key] = Tensor(dtype, shape, encoding, {"value": Part(dtype, shape)})
            continue
        found = tensors[mapped[key]]
        if tuple(found.shape) != shape:
            raise UnsupportedInput(f"{mapped[key]} has shape {found.shape}, the model needs {shape}")
        if found.logical_dtype == dtype:
            part = Part(dtype, shape, source=PartSource("source", component, mapped[key], "value"))
            rows[key] = Tensor(dtype, shape, encoding, {"value": part})
            continue
        if value.numel() * 4 > _CAST_LIMIT or (found.logical_dtype, dtype) != ("f32", "bf16"):
            raise UnsupportedInput(f"{mapped[key]} is {found.logical_dtype}; the model needs {dtype}")
        written[key] = mapped[key]
        rows[key] = Tensor(dtype, shape, encoding, {"value": Part(dtype, shape)})
    config = canonical_json.encode({
        "model_config": model_config,
        "files": {name: _blob(raw) for name, raw in sorted(members.items())},
    })
    definition = Derivation(
        sources={"source": capability},
        targets={COMPONENT: Target(source="source", source_component=component, drop=tuple(tensors), add=rows)},
        configs={COMPONENT: DerivedConfig("add")},
        order=tuple((COMPONENT, key) for key in expected),
    )
    with ctx.output("model").open(definition) as transaction:
        if transaction.receipt is not None:
            return ctx.adopt_model(transaction.receipt)
        completed = set(transaction.completed_parts())
        for key, payload in written.items():
            ctx.raise_if_cancelled()
            if (COMPONENT, key, "value") in completed:
                continue
            if isinstance(payload, str):
                raw = bytearray(expected[key].numel() * 4)
                transaction.source_read_into("source", component, payload, "value", 0, memoryview(raw))
                payload = torch.from_numpy(np.frombuffer(raw, dtype="<f4").copy()).to(torch.bfloat16).view(torch.uint8).numpy().tobytes()
            transaction.add_part(COMPONENT, key, "value", payload)
            transaction.checkpoint()
        if COMPONENT not in set(transaction.completed_configs()):
            transaction.add_config(COMPONENT, config)
            transaction.checkpoint()
        tel.log("describer model prepared", inherited=len(rows) - len(written), written=len(written))
        return ctx.adopt_model(transaction.commit())
