"""HunyuanImage 3.0 Instruct on Runtime: Tencent's model (`tencent/`, ported from
`Tencent-Hunyuan/HunyuanImage-3.0` at 6e9113a6) built as one component and filled from the checkpoint as
uploaded; the tokenizer is the checkpoint's own.

One 80B mixture-of-experts decoder writes the chain of thought and the rewritten prompt token by token,
then denoises the image's latents in the same transformer. Its 64 experts per layer are separate linears,
so Runtime's int8 and nvfp4 lanes take them like any other.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import torch
from cozy_runtime.author import Config, Loader, Model, uncaptured, uses_components
from PIL import Image
from tokenizers import Tokenizer

from .tencent.configuration_hunyuan_image_3 import HunyuanImage3Config
from .tencent.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
from .tencent.system_prompt import get_system_prompt
from .tencent.tokenization_hunyuan_image_3 import HunyuanImage3TokenizerFast

COMPONENT = "model"

#: Tencent's `generation_config.json`, the same in both checkpoints but for the step count, which each
#: function sets (`diff_infer_steps`: 50 Instruct, 8 Distil).
GENERATION = {
    "do_sample": True,
    "top_k": 1024,
    "top_p": 0.95,
    "temperature": 0.6,
    "max_length": 22800,
    "sequence_template": "instruct",
    "diff_guidance_scale": 2.5,
    "flow_shift": 3.0,
    "use_system_prompt": "en_unified",
    "drop_think": False,
    "bot_task": "think_recaption",
    "max_new_tokens": 2048,
}
#: `prompt_upsampling` on: Tencent's default stage, think then recaption; off: straight to the image.
UPSAMPLED, DIRECT = "think_recaption", "image"
_RECAPTION = re.compile(r"<recaption>(.*?)(?:</recaption>|$)", re.DOTALL)


class Graph:
    def __init__(self, config: Config) -> None:
        # As Tencent's `from_pretrained(torch_dtype="auto")` builds it: every module in bfloat16 (the
        # VAE's float32 weights included) but the experts' routers, which are float32.
        model = HunyuanImage3ForCausalMM(HunyuanImage3Config(**config.mapping()))
        model = model.to(torch.bfloat16).eval()
        for layer in model.model.layers:
            layer.mlp.gate.wg.float()
        self.components: dict[str, torch.nn.Module] = {COMPONENT: model}


def _tokenizer(read: Callable[[str], bytes]) -> HunyuanImage3TokenizerFast:
    config = json.loads(read("tokenizer_config.json"))
    config.pop("tokenizer_class", None)
    return HunyuanImage3TokenizerFast(tokenizer_object=Tokenizer.from_str(read("tokenizer.json").decode()), **config)


def _system(bot_task: str) -> str:
    return (get_system_prompt(GENERATION["use_system_prompt"], bot_task) or "").strip()


def caption(cot: str | None, prompt: str) -> str:
    """The caption the image was made from: the model's rewrite when it wrote one."""
    if cot is None:
        return prompt
    found = _RECAPTION.search(cot)
    return found.group(1).strip() if found else prompt


@uncaptured(COMPONENT)
class HunyuanImage3(Model[Graph], encoded_leaves="accept"):
    """Text to image, or an edit of up to three ordered references, with the thinking stage optional.
    Its routing and token loop wait on the host once per layer, so the component is never graph-recorded."""

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

    @uses_components(COMPONENT)
    def think(self, prompt: str, references: list[Image.Image], *, seed: int, on_token: Callable[[int], None],
              cancel: Callable[[], None]) -> str:
        """Tencent's think-then-recaption stage: the chain of thought and rewritten prompt the image is made from."""
        model = self.graph.components[COMPONENT]
        with torch.inference_mode():
            generator = torch.Generator(model.device).manual_seed(seed)
            return model.generate_cot(prompt, self._cond(references), _system(UPSAMPLED), UPSAMPLED, seed=seed,
                                      generator=generator, on_token=on_token, cancel=cancel)

    @uses_components(COMPONENT)
    def paint(self, prompt: str, references: list[Image.Image], cot: str | None, *, size: tuple[int, int],
              steps: int, guidance: float, seed: int, on_step: Callable[[int], None],
              cancel: Callable[[], None]) -> Image.Image:
        """Tencent's image stage at an explicit `size` (width, height), after `think` or without it."""
        model = self.graph.components[COMPONENT]
        width, height = size
        with torch.inference_mode():
            latents = model.denoise(prompt, self._cond(references), _system(UPSAMPLED if cot else DIRECT), cot,
                                    image_size=f"{height}x{width}", steps=steps, guidance_scale=guidance, seed=seed,
                                    on_step=on_step, cancel=cancel)
            return model.decode_latents(latents)
