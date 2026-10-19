"""A Transformers image-text model served from its as-is upload: the construction config is the
repository's `config.json`, and the tokenizer, processor and chat template are the checkpoint's assets."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

import torch
import transformers
from cozy_runtime import lm
from cozy_runtime.author import Config, Loader, Model, ModelAssets, uses_components
from PIL.Image import Image
from tokenizers import Tokenizer
from transformers import AutoConfig, AutoModelForImageTextToText, GenerationConfig, PreTrainedTokenizerFast
from transformers.initialization import no_init_weights
from transformers.video_utils import VideoMetadata

COMPONENT = "model"
DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}


def _construct(model_config: Mapping[str, Any], config: Config) -> torch.nn.Module:
    """Every parameter in its stored dtype: Qwen3.5 keeps `A_log` and its gated norms in f32, and
    Runtime fills stored bytes as they are."""
    with no_init_weights():
        model = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**model_config), dtype=torch.bfloat16)
    model.tie_weights()
    for name, parameter in model.named_parameters():
        stored = DTYPES.get(config.tensor_dtype(f"{COMPONENT}.{name}", default="bf16"))
        if stored is not None and parameter.dtype != stored:
            parameter.data = parameter.data.to(stored)
    return model.eval()


def _processor(assets: ModelAssets) -> Any:
    """The upstream processor from the checkpoint's files, parsed in memory."""
    names = set(assets.names())
    image = json.loads(assets.read("preprocessor_config.json"))
    processor_class = getattr(transformers, image.pop("processor_class"))
    image_processor = getattr(transformers, image.pop("image_processor_type"))(**image)
    extra = json.loads(assets.read("processor_config.json")) if "processor_config.json" in names else {}
    extra.pop("processor_class", None)
    if "video_preprocessor_config.json" in names:
        video = json.loads(assets.read("video_preprocessor_config.json"))
        video.pop("processor_class", None)
        extra["video_processor"] = getattr(transformers, video.pop("video_processor_type"))(**video)
    tokenizer_config = json.loads(assets.read("tokenizer_config.json"))
    for key in ("added_tokens_decoder", "tokenizer_file", "processor_class", "chat_template"):
        tokenizer_config.pop(key, None)
    named = tokenizer_config.pop("tokenizer_class", "")
    tokenizer_class = getattr(transformers, named, None) if named != "PreTrainedTokenizer" else None
    tokenizer = (tokenizer_class or PreTrainedTokenizerFast)(
        tokenizer_object=Tokenizer.from_str(assets.read("tokenizer.json").decode()), **tokenizer_config
    )
    if "chat_template.jinja" in names:
        chat = assets.read("chat_template.jinja").decode()
    else:
        chat = json.loads(assets.read("chat_template.json"))["chat_template"]
    return processor_class(image_processor=image_processor, tokenizer=tokenizer, chat_template=chat, **extra)


def _draft(model: torch.nn.Module, config: Config) -> Any:
    """Qwen3.5's multi-token-prediction head where the checkpoint keeps it (`mtp.*`), as part of
    the model's component: `lm.generate` drafts tokens with it and keeps those the model agrees
    with. None for a checkpoint without it."""
    text = model.config.get_text_config(decoder=True)
    if not getattr(text, "mtp_num_hidden_layers", 0) or not config.tensor_dtype(
        f"{COMPONENT}.mtp.fc.weight", default=""
    ):
        return None
    with no_init_weights():
        draft = lm.Draft(model.config, sys.modules[type(model).__module__])
    model.mtp = draft.mtp.to(torch.bfloat16)
    return draft.eval()


class Pipeline:
    def __init__(self, config: Config) -> None:
        model = _construct(config.mapping(), config)
        self.draft = _draft(model, config)
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model}


class Describer(Model[Pipeline], encoded_leaves="accept", min_bits=4, phase="decode"):
    """One greedy reply about one image. Runtime may serve encoded lanes' linears natively."""

    pipe: Pipeline
    processor: Any

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=Pipeline)
        self.processor = _processor(loader.assets)
        if "generation_config.json" in loader.assets.names():
            model = self.pipe.components[COMPONENT]
            model.generation_config = GenerationConfig(**json.loads(loader.assets.read("generation_config.json")))

    def execution_plan(self, plan: Any) -> str:
        return lm.install(plan, self.pipe.components[COMPONENT])

    def unload(self, loader: Loader) -> None:
        del self.pipe

    @uses_components(COMPONENT)
    def describe(self, images: list[Image], prompts: list[str], *, max_new_tokens: int) -> list[str]:
        """One greedy reply per image and its prompt, all at once: one prefill over the rows,
        left-padded, and one decode of them together."""
        texts = [self._chat(_messages(self.processor, prompt)) for prompt in prompts]
        replies = self._generate(self._inputs(text=texts, images=images), max_new_tokens)
        return [" ".join(reply.split()) for reply in replies]

    @uses_components(COMPONENT)
    def describe_video(self, frames: list[Image], seconds: list[float], prompt: str, *, max_new_tokens: int) -> str:
        """One greedy reply about frames watched as one video, each block stamped with its
        `<t seconds>` as the processor writes them. Line breaks are kept."""
        content = [{"type": "video"}, {"type": "text", "text": prompt}]
        # The processor stamps a block with frame index / fps: millisecond indices keep the seconds.
        metadata = VideoMetadata(
            total_num_frames=len(frames), fps=1000.0, frames_indices=[round(t * 1000) for t in seconds]
        )
        inputs = self._inputs(
            text=[self._chat([{"role": "user", "content": content}])], videos=[frames],
            video_metadata=[metadata], do_sample_frames=False,
        )
        (reply,) = self._generate(inputs, max_new_tokens)
        return reply.strip()

    @uses_components(COMPONENT)
    def write(self, prompts: list[str], *, max_new_tokens: int) -> list[str]:
        """One greedy reply per text prompt, all at once. Line breaks are kept."""
        texts = [self._chat([{"role": "user", "content": [{"type": "text", "text": p}]}]) for p in prompts]
        return [reply.strip() for reply in self._generate(self._inputs(text=texts), max_new_tokens)]

    def _chat(self, messages: list[dict[str, Any]]) -> str:
        text: str = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        return text

    def _inputs(self, **kwargs: Any) -> Any:
        model = self.pipe.components[COMPONENT]
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(padding=True, return_tensors="pt", **kwargs).to(model.device)
        for name in ("pixel_values", "pixel_values_videos"):
            if name in inputs:
                inputs[name] = inputs[name].to(model.dtype)
        return inputs

    def _generate(self, inputs: Any, max_new_tokens: int) -> list[str]:
        model, processor = self.pipe.components[COMPONENT], self.processor
        configured = model.generation_config.eos_token_id
        stops = [*(configured if isinstance(configured, list) else [configured]), processor.tokenizer.eos_token_id]
        stop = [token for token in dict.fromkeys(stops) if token is not None]
        out = lm.generate(
            model, inputs, max_new_tokens=max_new_tokens, stop=stop,
            pad=processor.tokenizer.pad_token_id or stop[0], draft=self.pipe.draft,
        )
        return [str(reply) for reply in processor.batch_decode(out, skip_special_tokens=True)]


def _messages(processor: Any, prompt: str) -> list[dict[str, Any]]:
    if type(processor).__name__ == "LlavaProcessor":
        return [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
