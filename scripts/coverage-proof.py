#!/usr/bin/env python3
"""Every tensor a package's checkpoint stores is one its construction loads, and every persistent
buffer it builds comes from the checkpoint (se-074), on the real cozytensors/1 headers in
`lane-headers/`:

    flux-3-klein/.venv/bin/python scripts/coverage-proof.py flux-3-klein

Each model is built as Runtime serves it (its serving substrate: weights on meta) and judged by
Runtime's own fit, less what its Model class declares (`unloaded=`, `derived=`). A load that
disagrees still serves and names the tensors in its warnings; a first-party package has none.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import tensorfs
from cozy_runtime.author._loader import Config, census
from cozy_runtime.author._model import uncovered
from cozy_runtime.internal import canonical, model_config
from cozy_runtime.internal.derive import Observations, serving_substrate
from cozy_runtime.internal.fill import Checkpoint, TensorRequirement, fit, tensorfs_requirement_dtype

ROOT = Path(__file__).resolve().parent.parent

#: package -> (module directory, [(header, "module:factory", "module:ModelClass")]): each factory
#: takes the header's construction config, as the class's `loader.construct` calls it.
MODELS = {
    "flux-3-klein": ("flux-3-klein", [
        ("flux-3-klein", "flux_3_klein.klein:KleinGraph", "flux_3_klein.klein:Klein"),
        ("flux-3-klein-upsampler", "flux_3_klein.upsampler:UpsamplerGraph", "flux_3_klein.upsampler:Upsampler"),
        ("qwen3.5-9b-mtp", "flux_3_klein.upsampler:DrafterGraph", "flux_3_klein.upsampler:Drafter"),
    ]),
    "sdxl": ("sdxl", [("sdxl", "sdxl:build_pipeline", "sdxl:SdxlModel")]),
    "anima": ("anima", [("anima", "anima.model:build_pipeline", "anima.model:AnimaModel")]),
    "seedvr2": ("seedvr2", [("seedvr2", "seedvr2.model:build", "seedvr2.model:RestorationModel")]),
    "qwen-image-2": ("qwen-image-2", [
        ("qwen-image-2", "qwen_image_2.model:build_qwen_image21", "qwen_image_2.model:QwenImage21Model"),
    ]),
    "character-describer": ("character-describer", [
        ("qwen3.5-9b", "character_describer.vlm:Pipeline", "character_describer.vlm:Describer"),
    ]),
    "birefnet": ("birefnet", [("birefnet", "birefnet.model:build_birefnet", "birefnet.model:BackgroundRemovalModel")]),
}


def _named(path: str) -> object:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def gaps(header: str, factory: str, model: str) -> tuple[list[str], list[str]]:
    """(stored tensors the construction does not load, persistent buffers the header lacks), less
    what `model` declares."""
    checkpoint = Checkpoint(tensorfs.open(ROOT / "scripts" / "lane-headers" / f"{header}.cozytensors",
                                          local_files_only=True))
    document = canonical.parse_canonical(model_config.construction_config(checkpoint.configs()))
    with serving_substrate("cpu", Observations()):
        built = _named(factory)(Config(document, f"{header}.config"))  # type: ignore[operator]
    walked = census(built)
    requirements = [
        TensorRequirement(d.component, d.key[len(d.component) + 1 :], list(d.spec.shape),
                          tensorfs_requirement_dtype(d.spec.dtype), buffer=d.buffer)
        for d in walked.destinations
    ]  # fmt: skip
    verdict = fit(requirements, checkpoint, encoded_leaves=True)
    assert verdict["ok"], f"{header}: {verdict['code']} {verdict['detail']}"
    return uncovered(_named(model), verdict["ignored"], verdict["unstored"])  # type: ignore[arg-type]


def main(package: str) -> None:
    if package not in MODELS:
        print(f"{package}: no recorded header")
        return
    directory, models = MODELS[package]
    sys.path.insert(0, str(ROOT / directory))
    for header, factory, model in models:
        ignored, unstored = gaps(header, factory, model)
        assert not ignored, f"{factory} on {header} does not load {len(ignored)}: {ignored[:8]}"
        assert not unstored, f"{factory} on {header}: persistent buffers not stored: {unstored[:8]}"
        print(f"{package} {factory} on {header}: every stored tensor loads, every buffer is stored")
    print(f"{package} coverage PASS")


if __name__ == "__main__":
    main(sys.argv[1])
