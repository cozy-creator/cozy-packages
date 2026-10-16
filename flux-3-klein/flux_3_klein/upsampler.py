"""FLUX.3 Klein's prompt upsampler (a Qwen3.5-9B fine-tune), served from its as-is upload: the
construction config is BFL's `config.json`, and the tokenizer, processor and chat template are the
checkpoint's assets.

Inference follows BFL's `flux3/pup.py`: greedy decoding, presence penalty 1.5 for text-to-image and 0
for edits, references resized to the upsampler's budget and round-tripped through JPEG q92.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping
from importlib.resources import files
from typing import Any

import torch
import transformers
from cozy_runtime import lm
from cozy_runtime.author import Config, Loader, Model, ModelAssets, UnsupportedInput, uses_components
from PIL import Image, ImageOps
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedTokenizerFast
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


def _construct(model_config: Mapping[str, Any]) -> torch.nn.Module:
    with no_init_weights():
        model = AutoModelForImageTextToText.from_config(
            AutoConfig.for_model(**model_config), dtype=torch.bfloat16
        )
    model.tie_weights()
    return model.eval()


def _processor(assets: ModelAssets) -> Any:
    """BFL's processor from the checkpoint's files, parsed in memory."""
    processor = json.loads(assets.read("processor_config.json"))
    image = processor["image_processor"]
    video = processor["video_processor"]
    tokenizer_config = json.loads(assets.read("tokenizer_config.json"))
    for key in ("added_tokens_decoder", "tokenizer_file", "processor_class", "tokenizer_class",
                "chat_template", "is_local", "local_files_only", "backend"):
        tokenizer_config.pop(key, None)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer.json").decode()), **tokenizer_config
    )
    return getattr(transformers, processor["processor_class"])(
        image_processor=getattr(transformers, image.pop("image_processor_type"))(**image),
        video_processor=getattr(transformers, video.pop("video_processor_type"))(**video),
        tokenizer=tokenizer,
        chat_template=assets.read("chat_template.jinja").decode(),
    )


class UpsamplerGraph:
    def __init__(self, config: Config) -> None:
        self.components: dict[str, torch.nn.Module] = {COMPONENT: _construct(config.mapping())}


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


class Upsampler(Model[UpsamplerGraph], encoded_leaves="accept"):
    """Rewrites a short request, and its references when editing, into the caption Klein wants."""

    graph: UpsamplerGraph
    processor: Any
    #: BFL's system prompts (`system_prompts.json` of its repository), by task.
    system_prompts: dict[str, str]

    def execution_plan(self, plan: Any) -> str:
        return lm.install(plan, self.graph.components[COMPONENT])

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(UpsamplerGraph, factory=UpsamplerGraph)
        self.processor = _processor(loader.assets)
        self.system_prompts = json.loads(files(__package__).joinpath("system_prompts.json").read_bytes())

    @uses_components(COMPONENT, memoize=True, memo_dependencies=(policy_reference,))
    def upsample(self, prompt: str, images: list[Image.Image], canvas: str | None) -> str:
        if len(images) > MAX_REFERENCES:
            raise UnsupportedInput(f"the upsampler takes at most {MAX_REFERENCES} references")
        model, processor = self.graph.components[COMPONENT], self.processor
        task = "i2i" if images else "t2i"
        if images:
            references = [policy_reference(image, len(images)) for image in images]
            text = I2I_SCAFFOLD.format(instruction=prompt)
            content: Any = [{"type": "image"} for _ in references] + [{"type": "text", "text": text}]
        else:
            references = []
            content = f"TARGET CANVAS: {canvas}\n{prompt}" if canvas else prompt
        messages = [
            {"role": "system", "content": self.system_prompts[task]},
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
        out = lm.generate(
            model, inputs, max_new_tokens=MAX_NEW_TOKENS, stop=stops, pad=tokenizer.pad_token_id,
            presence_penalty=PRESENCE_PENALTY[task],
        )
        caption = str(tokenizer.decode(out[0], skip_special_tokens=True)).strip()
        if not caption:
            raise UnsupportedInput("the prompt upsampler returned an empty caption", code="output_integrity")
        return caption
