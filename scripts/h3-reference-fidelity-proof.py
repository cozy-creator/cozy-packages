#!/usr/bin/env python
"""CPU-only proof of per-image geometry through the actual Diffusers H3 setup.

Constructs production components on meta, then runs only media setup and processor
patchification. No model forward, weights, accelerator, network, or test framework.
"""

from __future__ import annotations

import gc
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from cozy_runtime.author import Config, Image, canonical_json
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "minimax-h3"))
sys.path.insert(0, str(ROOT / "minimax-h3-tools" / "src"))

from cozy_runtime.models.minimax_h3.official import (  # noqa: E402
    REFERENCE_IMAGE_SHORT_EDGE,
    OfficialH3Pipeline,
    _ScopedPipeline,
    frames_for,
    reference_image_size,
    reference_image_vision_tokens,
    supported_durations,
    supported_steps,
)
from h3_tables.model_config import dual_full_config, parse_production_config  # noqa: E402


class Cancelled(BaseException):
    """An invocation discarded after actual setup, before any weighted stage."""


class RecordingVAE:
    """Capture the actual encoder boundary; return synthetic, unweighted posterior data."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.device = torch.device("cpu")
        self.input_shapes: list[tuple[int, ...]] = []

    def encode(self, pixels: Any, *, return_dict: bool) -> tuple[Any]:
        assert return_dict is False
        self.input_shapes.append(tuple(pixels.shape))
        height, width = pixels.shape[-2:]
        latent = torch.zeros(1, len(self.config.latents_mean), 1, height // 16, width // 16)
        return (SimpleNamespace(sample=lambda generator: latent),)


def meta_pipeline() -> OfficialH3Pipeline:
    source = ROOT / "minimax-h3-tools" / "src" / "h3_tables" / "assets" / "model-config.json"
    config = canonical_json.decode(dual_full_config(parse_production_config(source.read_bytes())))
    with torch.device("meta"):
        return OfficialH3Pipeline(Config(config))


def picture(width: int, height: int, *, rgba: bool = False) -> Image:
    channels = 4 if rgba else 3
    pixels = np.arange(width * height * channels, dtype=np.uint8)
    return PILImage.fromarray(pixels.reshape(height, width, channels))


def digest(image: Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def start(
    pipe: OfficialH3Pipeline, references: list[Any], edges: int | tuple[int, ...] = 2048
) -> Any:
    if isinstance(edges, int):
        edges = tuple(
            edges for reference in references if isinstance(reference, MiniMaxH3ImageReference)
        )
    return pipe.start_ref2va(
        prompt="<Picture 1> and <Picture 2> beside <Picture 3>.",
        references=references,
        generator=torch.Generator().manual_seed(7),
        steps=min(supported_steps()),
        frames=frames_for(min(supported_durations())),
        reference_image_short_edges=edges,
    )


def main() -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=''"
    assert Image is PILImage.Image
    pipe = meta_pipeline()
    shared = pipe._pipes["ref2va"]
    original_config = dict(shared.config)
    processor = shared.image_processor
    original_processor_config = dict(processor.config)
    source_images = [picture(381, 509, rgba=True), picture(960, 640), picture(513, 513)]
    original_sources = [(image.mode, image.size, digest(image)) for image in source_images]
    edges = (256, 1024, 2048)
    references = [pipe.image_reference(image) for image in source_images]
    assert all(isinstance(reference, MiniMaxH3ImageReference) for reference in references)
    state = start(pipe, references, edges)
    actual_sizes = [entry.image.size for entry in state.normalized_references]
    assert actual_sizes == [(256, 352), (1536, 1024), (2048, 2048)]
    for source, edge, reference, normalized in zip(
        source_images, edges, references, state.normalized_references, strict=True
    ):
        assert reference.image.size == source.size
        assert normalized.image.mode == "RGB"
        assert normalized.image.size == reference_image_size(source.width, source.height, edge)
        assert normalized.image.width * normalized.image.height // 1024 == (
            reference_image_vision_tokens(source.width, source.height, edge)
        )

    encoder = pipe._blocks["ref2va"].sub_blocks["text_encoder"]
    features, image_tokens, video_tokens, timestamps = encoder._gather_vision_features(
        shared.processor, state.normalized_references, shared.fps
    )
    grids = features["image_grid_thw"].tolist()
    assert grids == [[1, 22, 16], [1, 64, 96], [1, 128, 128]]
    assert image_tokens == [88, 1536, 4096]
    assert image_tokens == [
        reference_image_vision_tokens(image.width, image.height, edge)
        for image, edge in zip(source_images, edges, strict=True)
    ]
    assert video_tokens == timestamps == []
    del features
    gc.collect()

    # Run the real VAE-encoder block up to encode(), without any weighted VAE forward.
    recording_vae = RecordingVAE(shared.vae.config)
    vae_scope = _ScopedPipeline(shared, recording_vae, overrides={"vae": recording_vae})
    pipe._blocks["ref2va"].sub_blocks["vae_encoder"](vae_scope, state)
    assert recording_vae.input_shapes == [
        (1, 3, 1, 352, 256),
        (1, 3, 1, 1024, 1536),
        (1, 3, 1, 2048, 2048),
    ]

    # Negative control: unchanged upstream setup really does upscale the small reference.
    unmarked = start(pipe, [MiniMaxH3ImageReference(image=references[0].image)])
    assert min(unmarked.normalized_references[0].image.size) == 2048
    assert unmarked.normalized_references[0].image.size != actual_sizes[0]

    # Default auto and request-global overrides match the previous NumPy input path exactly.
    baseline_pixels = np.asarray(source_images[0].convert("RGB"))
    defaults = []
    for edge in (256, 769, 1024, REFERENCE_IMAGE_SHORT_EDGE):
        baseline = start(pipe, [MiniMaxH3ImageReference(image=baseline_pixels)], edge)
        explicit = start(pipe, [pipe.image_reference(source_images[0])], edge)
        actual = explicit.normalized_references[0].image
        assert actual.size == baseline.normalized_references[0].image.size
        assert actual.tobytes() == baseline.normalized_references[0].image.tobytes()
        defaults.append({"short_edge": edge, "size": list(actual.size), "sha256": digest(actual)})
    automatic = start(pipe, [pipe.image_reference(source_images[0])])
    assert digest(automatic.normalized_references[0].image) == defaults[-1]["sha256"]
    boundary = picture(400, 100)
    boundary_baseline = start(pipe, [MiniMaxH3ImageReference(image=boundary)], 784)
    boundary_auto = start(
        pipe, [pipe.image_reference(boundary) for _ in range(3)], (784, 256, 2048)
    )
    assert boundary_auto.normalized_references[0].image.size == (3136, 768)
    boundary_sizes = [entry.image.size for entry in boundary_auto.normalized_references]
    assert boundary_sizes == [(3136, 768), (1024, 256), (8192, 2048)]
    assert boundary_auto.normalized_references[0].image.tobytes() == (
        boundary_baseline.normalized_references[0].image.tobytes()
    )
    # The same underlying source can occur twice with different edges and separate pixels.
    repeated = start(
        pipe,
        [pipe.image_reference(source_images[2]) for _ in range(2)],
        (256, 2048),
    )
    assert [entry.image.size for entry in repeated.normalized_references] == [
        (256, 256),
        (2048, 2048),
    ]
    assert repeated.normalized_references[0].image is not repeated.normalized_references[1].image
    assert [entry.image.size for entry in start(pipe, references, edges).normalized_references] == (
        actual_sizes
    )

    # Actual setup errors after a sized reference must not leave an override behind.
    malformed = MiniMaxH3ImageReference(image=np.zeros((2, 2, 4), dtype=np.uint8))
    try:
        start(pipe, [references[0], malformed], (256, 1024))
    except ValueError as exc:
        assert "RGB pixels" in str(exc)
    else:
        raise AssertionError("malformed reference did not fail upstream setup")

    block = pipe._blocks["ref2va"].sub_blocks["before_encode"]

    def cancelled_setup(components: Any, pending: Any) -> Any:
        block(components, pending)
        assert pending.normalized_references[0].image.size == actual_sizes[0]
        raise Cancelled

    pipe._blocks["ref2va"].sub_blocks["before_encode"] = cancelled_setup
    try:
        try:
            start(pipe, [references[0]], 256)
        except Cancelled:
            pass
        else:
            raise AssertionError("cancellation did not abandon setup")
    finally:
        pipe._blocks["ref2va"].sub_blocks["before_encode"] = block
    recovered = start(pipe, [MiniMaxH3ImageReference(image=baseline_pixels)])
    assert digest(recovered.normalized_references[0].image) == defaults[-1]["sha256"]

    # Only image geometry changes in mixed reference requests; media clocks remain upstream.
    audio = MiniMaxH3AudioReference(audio=torch.arange(3200).float()[None], sample_rate=32000)
    video = MiniMaxH3VideoReference(
        frames=np.arange(7 * 32 * 64 * 3, dtype=np.uint8).reshape(7, 32, 64, 3),
        fps=12.0,
        audio=torch.arange(3200).float()[None],
        sample_rate=32000,
    )
    mixed = start(pipe, [references[0], video, audio, references[1]], (256, 1024))
    previous = start(
        pipe,
        [
            MiniMaxH3ImageReference(image=baseline_pixels),
            video,
            audio,
            MiniMaxH3ImageReference(image=np.asarray(source_images[1])),
        ],
    )
    normalized = mixed.normalized_references
    assert [entry.kind for entry in normalized] == ["image", "video", "audio", "image"]
    assert [normalized[0].image.size, normalized[3].image.size] == actual_sizes[:2]
    assert normalized[1].fps == 24.0
    assert normalized[1].frames.shape[0] == 14
    assert np.array_equal(normalized[1].frames, previous.normalized_references[1].frames)
    for index in (1, 2):
        assert normalized[index].sample_rate == 32000
        assert normalized[index].audio.shape == (2, 3200)
        assert torch.equal(normalized[index].audio, previous.normalized_references[index].audio)
    _, actual_timestamps = encoder._sample_video_condition_frames(
        normalized[1].frames,
        shared.fps,
        encoder.video_sample_fps,
        shared.processor.video_processor.temporal_patch_size,
    )
    _, baseline_timestamps = encoder._sample_video_condition_frames(
        previous.normalized_references[1].frames,
        shared.fps,
        encoder.video_sample_fps,
        shared.processor.video_processor.temporal_patch_size,
    )
    assert actual_timestamps == baseline_timestamps
    assert dict(shared.config) == original_config
    assert shared.image_processor is processor
    assert dict(processor.config) == original_processor_config
    assert original_sources == [(image.mode, image.size, digest(image)) for image in source_images]

    print(
        json.dumps(
            {
                "status": "passed",
                "runtime_version": version("cozy-runtime"),
                "diffusers_version": version("diffusers"),
                "weights_loaded": False,
                "model_forward": False,
                "cuda_available": torch.cuda.is_available(),
                "threads": torch.get_num_threads(),
                "short_edges": list(edges),
                "normalized_sizes": [list(size) for size in actual_sizes],
                "image_grid_thw": grids,
                "vision_tokens": image_tokens,
                "actual_vae_encoder_input_shapes": recording_vae.input_shapes,
                "vae_posterior": "synthetic boundary recorder; no weighted VAE forward",
                "upstream_default_parity": defaults,
                "non_grid_global_edge_aspect_boundary_parity": True,
                "mixed_784_low_high_boundary_sizes": boundary_sizes,
                "mixed_video_frames": normalized[1].frames.shape[0],
                "mixed_video_fps": normalized[1].fps,
                "mixed_audio_sample_rate": normalized[2].sample_rate,
                "mixed_audio_samples": normalized[2].audio.shape[-1],
                "mixed_video_timestamps": actual_timestamps,
                "source_rgba_preserved": True,
                "repeat_error_cancel_isolation": True,
                "shared_config_and_processor_unchanged": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
