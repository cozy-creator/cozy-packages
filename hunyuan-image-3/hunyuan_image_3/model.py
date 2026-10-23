"""HunyuanImage 3.0 Instruct on Runtime: Tencent's model (`tencent/`, ported from
`Tencent-Hunyuan/HunyuanImage-3.0` at 6e9113a6) built as one component and filled from the checkpoint as
uploaded; the tokenizer is the checkpoint's own.

One 80B mixture-of-experts decoder writes the chain of thought and the rewritten prompt token by token,
then denoises the image's latents in the same transformer. Each of its 2,048 routed experts (64 in each of
32 layers, 37.7M parameters each: 93% of the model) is its own paging block, resident or paged on its own.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch
from cozy_runtime.author import Config, Loader, Model, uses_components
from PIL import Image
from tokenizers import Tokenizer

from .tencent.configuration_hunyuan_image_3 import HunyuanImage3Config
from .tencent.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
from .tencent.system_prompt import get_system_prompt
from .tencent.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast

COMPONENT = "model"

#: The text stage stops after this many tokens.
MAX_NEW_TOKENS = 2048
#: Both checkpoints' guidance: classifier-free for Instruct, the distilled value for Distil.
GUIDANCE = 2.5
#: Tencent's `generation_config.json`, the same in both checkpoints but for the step count, which each
#: function sets (`diff_infer_steps`: 8 Distil, 50 Instruct).
GENERATION: dict[str, object] = {
    "do_sample": True,
    "top_k": 1024,
    "top_p": 0.95,
    "temperature": 0.6,
    "max_length": 22800,
    "sequence_template": "instruct",
    "diff_guidance_scale": GUIDANCE,
    "flow_shift": 3.0,
    "use_system_prompt": "en_unified",
    "drop_think": False,
    "bot_task": "think_recaption",
    "max_new_tokens": MAX_NEW_TOKENS,
}
#: `prompt_upsampling` on: Tencent's default stage, think then recaption; off: straight to the image.
UPSAMPLED, DIRECT = "think_recaption", "image"
_RECAPTION = re.compile(r"<recaption>(.*?)(?:</recaption>|$)", re.DOTALL)
#: Expert caches the routing record replays (least recently used), in experts: 256 is ~9.7 GB at int8.
_CACHES = (256, 512, 1024)


class Graph:
    def __init__(self, config: Config) -> None:
        # As Tencent's `from_pretrained(torch_dtype="auto")` builds it: every module in bfloat16 (the
        # VAE's float32 weights included) but the experts' routers, which are float32.
        model = HunyuanImage3ForCausalMM(HunyuanImage3Config(**config.mapping()))
        model = model.to(torch.bfloat16).eval()
        for layer in model.model.layers:
            layer.mlp.gate.wg.float()
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model}


class Routing:
    """A stage's expert hits: per forward and layer, which of the 64 experts got tokens and how many, and how
    often an expert cache of each `_CACHES` size would have held the expert a forward needed."""

    def __init__(self) -> None:
        self.forwards = 0
        self.hit_per_layer: list[int] = []
        self.tokens_per_hit: list[int] = []
        self.touched: set[tuple[int, int]] = set()
        self.caches = {size: OrderedDict[tuple[int, int], None]() for size in _CACHES}
        self.cache_hits = dict.fromkeys(_CACHES, 0)

    def __call__(self, layer: int, counts: list[int]) -> None:
        self.forwards += layer == 0
        hit = [expert for expert, tokens in enumerate(counts) if tokens]
        self.hit_per_layer.append(len(hit))
        self.tokens_per_hit.extend(counts[expert] for expert in hit)
        for expert in hit:
            key = (layer, expert)
            self.touched.add(key)
            for size, cache in self.caches.items():
                if key in cache:
                    cache.move_to_end(key)
                    self.cache_hits[size] += 1
                else:
                    cache[key] = None
                    if len(cache) > size:
                        cache.popitem(last=False)

    def summary(self) -> dict[str, float]:
        """Means over forwards and layers; `working_set` counts distinct (layer, expert) pairs of the stage."""
        if not self.hit_per_layer:
            return {}
        uses = len(self.tokens_per_hit)
        found: dict[str, float] = {
            "forwards": self.forwards,
            "experts_hit_per_layer_mean": round(statistics.fmean(self.hit_per_layer), 2),
            "experts_hit_per_layer_min": min(self.hit_per_layer),
            "tokens_per_hit_expert_min": min(self.tokens_per_hit),
            "tokens_per_hit_expert_median": statistics.median(self.tokens_per_hit),
            "tokens_per_hit_expert_max": max(self.tokens_per_hit),
            "working_set": len(self.touched),
        }
        for size, hits in self.cache_hits.items():
            found[f"lru_{size}_hit_rate"] = round(hits / uses, 4)
        return found


def _tokenizer(read: Callable[[str], bytes]) -> HunyuanImage3TokenizerFast:
    config = json.loads(read("tokenizer_config.json"))
    config.pop("tokenizer_class", None)
    return HunyuanImage3TokenizerFast(tokenizer_object=Tokenizer.from_str(read("tokenizer.json").decode()), **config)


def _system(bot_task: str) -> str:
    return str(get_system_prompt(GENERATION["use_system_prompt"], bot_task) or "").strip()


def caption(cot: str | None, prompt: str) -> str:
    """The caption the image was made from: the model's rewrite when it wrote one."""
    if cot is None:
        return prompt
    found = _RECAPTION.search(cot)
    return found.group(1).strip() if found else prompt


class HunyuanImage3(Model[Graph], encoded_leaves="accept"):
    """Text to image, or an edit of up to three ordered references, with the thinking stage optional.
    Every forward takes the KV cache object, so Runtime runs it as written and records no CUDA graph."""

    graph: Graph

    def load(self, loader: Loader) -> None:
        self.graph = loader.construct(Graph, factory=Graph)
        model = self.graph.components[COMPONENT]
        model._tokenizer = _tokenizer(loader.assets.read)
        model.generation_config = SimpleNamespace(**GENERATION)

    def _cond(self, references: list[Image.Image]) -> Any:
        model = self.graph.components[COMPONENT]
        if not references:
            return None
        return [model.image_processor.build_cond_images(image_list=references, infer_align_image_size=True)]

    def _size(self, ratio_index: int) -> tuple[int, int]:
        size = self.graph.components[COMPONENT].image_processor.vae_reso_group[ratio_index]
        return size.width, size.height

    @contextmanager
    def _routing(self) -> Iterator[Routing]:
        routing = Routing()
        layers = [layer.mlp for layer in self.graph.components[COMPONENT].model.layers]
        for moe in layers:
            moe.trace = routing
        try:
            yield routing
        finally:
            for moe in layers:
                moe.trace = None

    @uses_components(COMPONENT)
    def think(self, prompt: str, references: list[Image.Image], *, seed: int, pick_size: bool,
              on_token: Callable[[int], None],
              cancel: Callable[[], None]) -> tuple[str, tuple[int, int] | None, dict[str, float]]:
        """Tencent's think-then-recaption stage: the chain of thought and rewritten prompt the image is made from,
        with `pick_size` the size (width, height) the model then picks (`image_size="auto"`), and its routing."""
        model = self.graph.components[COMPONENT]
        with torch.inference_mode(), self._routing() as routing:
            generator = torch.Generator(model.device).manual_seed(seed)
            cot, ratio = model.generate_cot(prompt, self._cond(references), _system(UPSAMPLED), UPSAMPLED, seed=seed,
                                            generator=generator, need_ratio=pick_size, on_token=on_token,
                                            cancel=cancel)
        return cot, None if ratio is None else self._size(ratio), routing.summary()

    @uses_components(COMPONENT)
    def pick_size(self, prompt: str, references: list[Image.Image], *, seed: int) -> tuple[int, int]:
        """The size (width, height) the model picks for a prompt it draws directly (`image_size="auto"`)."""
        model = self.graph.components[COMPONENT]
        with torch.inference_mode():
            return self._size(model.predict_ratio(prompt, self._cond(references), _system(DIRECT), seed=seed))

    @uses_components(COMPONENT)
    def paint(self, prompt: str, references: list[Image.Image], cot: str | None, *, size: tuple[int, int],
              steps: int, guidance: float, seed: int, on_step: Callable[[int], None],
              cancel: Callable[[], None]) -> tuple[Image.Image, dict[str, float]]:
        """Tencent's image stage at an explicit `size` (width, height), after `think` or without it, and its
        routing."""
        model = self.graph.components[COMPONENT]
        width, height = size
        with torch.inference_mode(), self._routing() as routing:
            latents = model.denoise(prompt, self._cond(references), _system(UPSAMPLED if cot else DIRECT), cot,
                                    image_size=f"{height}x{width}", steps=steps, guidance_scale=guidance, seed=seed,
                                    on_step=on_step, cancel=cancel)
            image: Image.Image = model.decode_latents(latents)
        return image, routing.summary()
