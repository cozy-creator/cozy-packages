"""FLUX.3 Klein's prompt upsampler (a Qwen3.5-9B fine-tune) and the recipe that prepares its checkpoint.

Inference follows BFL's `flux3/pup.py`: greedy decoding, presence penalty 1.5 for text-to-image and 0
for edits, references resized to the upsampler's budget and round-tripped through JPEG q92.
"""

from __future__ import annotations

import base64
import io
import json
import math
import zlib
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import msgspec
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
from PIL import Image, ImageOps
from tensorfs.derived import Config as DerivedConfig
from tensorfs.derived import Derivation, Part, PartSource, Target, Tensor
from tokenizers import Tokenizer
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    LogitsProcessorList,
    PreTrainedTokenizerFast,
)
from transformers.initialization import no_init_weights

COMPONENT = "model"
MAX_NEW_TOKENS = 3584
MAX_REFERENCES = 10
PRESENCE_PENALTY = {"t2i": 1.5, "i2i": 0.0}
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)
REFERENCE_MULTIPLE = 32
MAX_PIXELS_BY_REFS = {1: 1024 * 1024, 2: 724 * 724, **{n: 512 * 512 for n in range(3, 11)}}
DTYPES = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32", torch.int64: "i64"}


class ArtifactConfig(msgspec.Struct, frozen=True):
    model_config: dict[str, Any]
    files: dict[str, str]


def _blob(raw: bytes) -> str:
    # Config refuses strings containing `://`, which tokenizer merges contain.
    return base64.urlsafe_b64encode(zlib.compress(raw, 6)).decode("ascii")


def _unblob(value: str) -> bytes:
    return zlib.decompress(base64.urlsafe_b64decode(value))


def _construct(model_config: Mapping[str, Any]) -> torch.nn.Module:
    with no_init_weights():
        model = AutoModelForImageTextToText.from_config(
            AutoConfig.for_model(**model_config), dtype=torch.bfloat16
        )
    model.tie_weights()
    # Runtime fills only stored bytes, so every buffer is part of the checkpoint.
    for module in model.modules():
        module._non_persistent_buffers_set.clear()
    return model.eval()


def _processor(members: Mapping[str, bytes]) -> Any:
    """BFL's processor, built from bytes: model loading never touches the filesystem."""
    processor = json.loads(members["processor_config.json"])
    image = processor["image_processor"]
    video = processor["video_processor"]
    tokenizer_config = json.loads(members["tokenizer_config.json"])
    for key in ("added_tokens_decoder", "tokenizer_file", "processor_class", "tokenizer_class",
                "chat_template", "is_local", "local_files_only", "backend"):
        tokenizer_config.pop(key, None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(members["tokenizer.json"].decode()), **tokenizer_config
    )
    return getattr(transformers, processor["processor_class"])(
        image_processor=getattr(transformers, image.pop("image_processor_type"))(**image),
        video_processor=getattr(transformers, video.pop("video_processor_type"))(**video),
        tokenizer=tokenizer,
        chat_template=members["chat_template.jinja"].decode(),
    )


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        artifact = msgspec.convert(config.mapping(), ArtifactConfig)
        members = {name: _unblob(value) for name, value in artifact.files.items()}
        self.components: dict[str, torch.nn.Module] = {COMPONENT: _construct(artifact.model_config)}
        self.processor = _processor(members)
        self.system_prompts: dict[str, str] = json.loads(members["system_prompts.json"])


class _PresencePenalty:
    """vLLM's presence penalty: subtract `penalty` from every token already generated."""

    def __init__(self, prompt_length: int, penalty: float) -> None:
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1]:
            seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, generated, True)
            scores = scores - self.penalty * seen.to(scores.dtype)
        return scores


def _floor(value: float) -> int:
    return max(REFERENCE_MULTIPLE, int(math.floor(value / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE)


def reference_size(width: int, height: int, n_refs: int) -> tuple[int, int]:
    cap = MAX_PIXELS_BY_REFS[n_refs]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = _floor(width / beta), _floor(height / beta)
    if w * h > cap:
        if w == REFERENCE_MULTIPLE:
            h = _floor(cap / w)
        else:
            w = _floor(cap / h)
    return w, h


def policy_reference(image: Image.Image, n_refs: int) -> Image.Image:
    """RGB over white, LANCZOS to the size budget, JPEG q92: what the upsampler was trained on."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = reference_size(image.width, image.height, n_refs)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


class Upsampler(Model[UpsamplerGraph]):
    """Rewrites a short request, and its references when editing, into the caption Klein wants."""

    graph: UpsamplerGraph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph, factory=UpsamplerGraph)

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(policy_reference, _PresencePenalty))
    def upsample(self, prompt: str, images: list[Image.Image], canvas: str | None) -> str:
        if len(images) > MAX_REFERENCES:
            raise UnsupportedInput(f"the upsampler takes at most {MAX_REFERENCES} references")
        model, processor = self.graph.components[COMPONENT], self.graph.processor
        task = "i2i" if images else "t2i"
        if images:
            references = [policy_reference(image, len(images)) for image in images]
            text = I2I_SCAFFOLD.format(instruction=prompt)
            content: Any = [{"type": "image"} for _ in references] + [{"type": "text", "text": text}]
        else:
            references = []
            content = f"TARGET CANVAS: {canvas}\n{prompt}" if canvas else prompt
        messages = [
            {"role": "system", "content": self.graph.system_prompts[task]},
            {"role": "user", "content": content},
        ]
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = processor(
            text=[chat], images=references or None, return_tensors="pt"
        ).to(next(model.parameters()).device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        tokenizer = processor.tokenizer
        stops = [tokenizer.convert_tokens_to_ids(token) for token in STOP_TOKENS]
        length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([_PresencePenalty(length, PRESENCE_PENALTY[task])]),
            )
        caption = str(tokenizer.decode(out[0, length:], skip_special_tokens=True)).strip()
        if not caption:
            raise UnsupportedInput("the prompt upsampler returned an empty caption", code="output_integrity")
        return caption


def prepare(source: QuantizationSource, ctx: Context, tel: Telemetry) -> ModelArtifact:
    """BFL's raw upsampler weights with its configuration, tokenizer, processor and system prompts,
    which ship in `upsampler/` because a raw upload carries none of them."""
    root = files(__package__).joinpath("upsampler")
    members = {entry.name: entry.read_bytes() for entry in root.iterdir() if entry.is_file()}
    model_config = json.loads(members.pop("config.json"))
    with tel.stage("constructing the reference model"):
        model = _construct(model_config)
    expected = model.state_dict()
    capability = ctx.tensorfs_source(source)
    inspection = capability.inspect()
    if len(inspection.components) != 1:
        raise UnsupportedInput("the raw upsampler source must hold exactly one component")
    ((component, tensors),) = inspection.components.items()
    rows: dict[str, Tensor] = {}
    written: dict[str, bytes] = {}
    encoding = next(iter(tensors.values())).encoding
    for key, value in expected.items():
        dtype, shape = DTYPES[value.dtype], tuple(value.shape)
        found = tensors.get(key)
        if found is None:
            # A constructor buffer (rotary frequencies): its constructed value is stored.
            if key in dict(model.named_parameters()):
                raise UnsupportedInput(f"the upsampler source lacks the parameter {key}")
            written[key] = value.contiguous().view(torch.uint8).numpy().tobytes()
            rows[key] = Tensor(dtype, shape, encoding, {"value": Part(dtype, shape)})
            continue
        if tuple(found.shape) != shape or found.logical_dtype != dtype:
            raise UnsupportedInput(f"{key} is {found.logical_dtype}{found.shape}; the model needs {dtype}{shape}")
        part = Part(dtype, shape, source=PartSource("source", component, key, "value"))
        rows[key] = Tensor(dtype, shape, encoding, {"value": part})
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
            if (COMPONENT, key, "value") not in completed:
                transaction.add_part(COMPONENT, key, "value", payload)
                transaction.checkpoint()
        if COMPONENT not in set(transaction.completed_configs()):
            transaction.add_config(COMPONENT, config)
            transaction.checkpoint()
        tel.log("upsampler prepared", inherited=len(rows) - len(written), written=len(written))
        return ctx.adopt_model(transaction.commit())
