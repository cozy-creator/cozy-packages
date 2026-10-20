#!/usr/bin/env python
"""Real tiny Qwen modules: the package's staged model agrees exactly with Diffusers' pipeline.

Run with the package's own interpreter. The two parts run in separate processes: an executor
owns a fresh Python process, and its activity observer is process state.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
import tensorfs
from cozy_runtime.author import Config, ModelAssets
from cozy_runtime.author._activity import event_loop, observing
from cozy_runtime.author.fakes import granted_input, invoke_with_fakes
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen-image-2"))

from qwen_image_2 import app  # noqa: E402
from qwen_image_2.model import (  # noqa: E402
    QwenImage21Model,
    _StagedPipeline,
    build_processor,
    build_qwen_image21,
)


def photo(path: Path, size: tuple[int, int], orientation: int = 1) -> Path:
    exif = Image.Exif()
    exif[0x0112] = orientation
    Image.linear_gradient("L").resize(size).convert("RGB").save(path, exif=exif, quality=80)
    return path


def shapes(model: QwenImage21Model, tmp_path: Path) -> None:
    """generate_image through the real kernel on real JPEGs: an omitted aspect ratio takes the
    last reference's, an omitted tier the one nearest its pixels, and explicit values win."""
    square = photo(tmp_path / "square.jpg", (512, 512))
    cases: list[tuple[list[Path], dict[str, object], tuple[int, int]]] = [
        ([square, photo(tmp_path / "3x2.jpg", (1200, 800))], {}, (1248, 832)),
        # Stored landscape, EXIF-rotated upright: the output follows the displayed 9:16.
        ([photo(tmp_path / "9x16.jpg", (1280, 720), orientation=6)], {}, (768, 1376)),
        ([photo(tmp_path / "small.jpg", (640, 480))], {}, (1184, 896)),
        ([photo(tmp_path / "12mp.jpg", (4000, 3000))], {}, (2368, 1760)),
        ([tmp_path / "3x2.jpg"], {"megapixels": 2}, (1760, 1184)),
        ([tmp_path / "3x2.jpg"], {"aspect_ratio": "1:1", "megapixels": 2}, (1440, 1440)),
        ([], {"aspect_ratio": "7:3", "megapixels": 1}, (1568, 672)),
        # Past the rotary range: the nearest admitted size, reported as a clamp.
        ([], {"aspect_ratio": "5000:1", "megapixels": 1}, (32768, 32)),
        ([], {}, (2048, 2048)),
    ]
    for index, (paths, given, expected) in enumerate(cases):
        grants = {
            f"reference_images.{n}": granted_input(path, input_id=f"reference_images.{n}", order=n)
            for n, path in enumerate(paths)
        }
        wire = {"prompt": "a person", "steps": 1, "seed": 7, **given}
        wire["reference_images"] = [grant.digest for grant in grants.values()]
        result = invoke_with_fakes(
            app.get("generate_image"),
            wire,
            request_id=f"shape-{index}",
            spool=tmp_path / f"shape-{index}",
            models={"model": model},
            assets=grants,
        )
        image = result.result
        assert (image.width, image.height) == expected, (paths, given, image)
        with Image.open(tmp_path / f"shape-{index}" / "image-0001") as png:
            assert (png.format, png.size) == ("PNG", expected), (png.format, png.size)
        clamps = [(row.kind, row.field, row.requested) for row in result.adjustments]
        assert clamps == ([("clamp", "aspect_ratio", "5000:1")] if expected == (32768, 32) else [])


def staged(tmp_path: Path) -> None:
    geometry = {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2}
    image_config = {**geometry, "image_processor_type": "Qwen2VLImageProcessorFast"}
    config = {
        "processor": image_config,
        "transformer": dict(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=16,
            mlp_ratio=2,
            axes_dims_rope=[4, 6, 6],
        ),
        "vae": dict(
            base_dim=4,
            decoder_base_dim=4,
            z_dim=8,
            dim_mult=[1] * 5,
            num_res_blocks=1,
            attn_scales=[],
            temperal_downsample=[False, True, True, True],
            latents_mean=[0.0] * 8,
            latents_std=[1.0] * 8,
        ),
        "text_encoder": dict(
            # The tiny vocabulary's special tokens, so reference images reach the encoder.
            image_token_id=2,
            video_token_id=5,
            vision_start_token_id=6,
            vision_end_token_id=7,
            text_config=dict(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                rope_parameters=dict(rope_type="default", rope_theta=1e6, mrope_section=[1, 1, 2]),
            ),
            vision_config=dict(
                depth=1,
                hidden_size=16,
                intermediate_size=16,
                num_heads=2,
                out_hidden_size=16,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
                num_position_embeddings=64,
                deepstack_visual_indexes=[0],
            ),
        ),
        "scheduler": dict(FlowMatchEulerDiscreteScheduler().config),
    }
    torch.manual_seed(53)
    graph = build_qwen_image21(Config(config))
    # Native serving fills values; this real tiny fixture initializes its own values.
    for root in graph.components.values():
        for p in root.parameters():
            with torch.no_grad():
                p.normal_(0, 0.02)
        root.eval()
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|image_pad|>": 2,
        "<|im_start|>": 3,
        "<|im_end|>": 4,
        "<|video_pad|>": 5,
        "<|vision_start|>": 6,
        "<|vision_end|>": 7,
    }
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=t,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=[
            "<|image_pad|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ],
    )
    chat = (
        '{% for message in messages %}{{"<|im_start|>" + message["role"] + "\\n"}}'
        '{% for item in message["content"] %}{{item["text"]}}{% endfor %}'
        '{{"<|im_end|>\\n"}}{% endfor %}'
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=1024, max_pixels=1 << 21
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tok,
        chat_template=chat,
    )
    tokenizer_config = {"unk_token": "[UNK]", "pad_token": "[PAD]", "tokenizer_class": "X"}
    files = {
        "tokenizer.json": tok.backend_tokenizer.to_str().encode(),
        "tokenizer_config.json": json.dumps(tokenizer_config).encode(),
        "video_preprocessor_config.json": json.dumps(geometry).encode(),
        "chat_template.jinja": chat.encode(),
    }

    def assets(files: dict[str, bytes]) -> ModelAssets:
        named = {f"processor/{name}": raw for name, raw in files.items()}
        return ModelAssets(named, lambda _operation: None)

    def native_processor(
        name: str, configuration: dict[str, Any], config_file: str, files: dict[str, bytes]
    ) -> Any:
        """A real native component keeps its semantic config out of its assets."""
        store = tmp_path / f"processor-{name}"
        with tensorfs.Writer(cache_dir=store) as writer:
            component = writer.component(
                "processor", cls="Qwen3VLProcessor", config=configuration,
                config_file=config_file,
            )
            for filename, raw in files.items():
                component.asset(filename, raw)
            checkpoint = writer.commit()
        with tensorfs.open(checkpoint.id, cache_dir=store, local_files_only=True) as opened:
            component = opened.components["processor"]
            assert component.config_file == config_file
            assert dict(component.config) == configuration
            assert config_file not in component.assets
            supporting = {filename: component.asset(filename) for filename in component.assets}
            constructor = Config({"processor": dict(component.config)})
            return build_processor(constructor.section("processor").mapping(), assets(supporting))

    assert graph.processor_config == image_config
    rebuilt = native_processor("flat", graph.processor_config, "preprocessor_config.json", files)
    assert rebuilt.tokenizer.encode("blue jacket") == processor.tokenizer.encode("blue jacket")
    assert rebuilt.chat_template == chat
    # Turbo's Transformers 5 layout: one processor_config.json and the new tokenizer_config keys.
    turbo_files = {
        "tokenizer.json": files["tokenizer.json"],
        "tokenizer_config.json": json.dumps(
            {
                **tokenizer_config,
                "backend": "tokenizers",
                "is_local": True,
                "extra_special_tokens": [],
            }
        ).encode(),
        "chat_template.jinja": chat.encode(),
    }
    nested_config = {
        "image_processor": image_config, "video_processor": geometry, "processor_class": "X"
    }
    turbo = native_processor("nested", nested_config, "processor_config.json", turbo_files)
    assert turbo.tokenizer.encode("blue jacket") == processor.tokenizer.encode("blue jacket")
    assert turbo.to_dict() == rebuilt.to_dict()
    try:
        build_processor(
            graph.processor_config,
            assets({k: v for k, v in files.items() if k != "chat_template.jinja"}),
        )
    except Exception as missing:
        assert "chat_template.jinja" in str(missing), missing
    else:
        raise AssertionError("a checkpoint without its chat template was accepted")
    # Nonpersistent RoPE frequencies are rebuilt by official loading. Saving our exact
    # tiny weights and reloading them tests the construction boundary independently.
    encoder = graph.components["text_encoder"]
    encoder.save_pretrained(tmp_path)
    upstream_encoder = type(encoder).from_pretrained(
        tmp_path, dtype=torch.bfloat16, local_files_only=True
    )
    actual_buffers = dict(encoder.named_buffers())
    for name, expected_buffer in upstream_encoder.named_buffers():
        assert actual_buffers[name].dtype == expected_buffer.dtype == torch.float32
        torch.testing.assert_close(actual_buffers[name], expected_buffer, rtol=0, atol=0)
    assert {value.dtype for value in encoder.parameters()} == {torch.bfloat16}
    model = QwenImage21Model.for_test(graph=graph, processor=processor)
    frames: list[tuple[int, bool, bool, bool]] = []

    def observed(sequence: int, active: bool, known: bool, finished: bool) -> None:
        frames.append((sequence, active, known, finished))

    for _ in range(2):
        frames.clear()
        with observing(observed):
            embeds, mask = model.encode("a person in a blue jacket")
            assert embeds.shape[-1] == 16
            latents = model.denoise(
                embeds,
                mask,
                width=32,
                height=32,
                steps=2,
                seed=44,
                on_step=lambda i: None,
                cancel=lambda: None,
            )
            actual = model.decode(latents, width=32, height=32)
        assert frames[-1][1:] == (False, True, True), frames
    model.harness.assert_scopes("encode", "denoise", "decode", "encode", "denoise", "decode")
    pipe = QwenImage21Pipeline(
        **{**graph.components, "text_encoder": upstream_encoder},
        processor=processor,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config["scheduler"]),
    )
    expected = pipe(
        prompt="a person in a blue jacket",
        width=32,
        height=32,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(44),
        output_type="pt",
    ).images
    # Without the prefix KV cache (a card too small to hold it) the image is the same.
    uncached = pipe(
        prompt="a person in a blue jacket",
        width=32,
        height=32,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(44),
        output_type="pt",
        use_kv_cache=False,
    ).images
    torch.testing.assert_close(uncached, expected, rtol=0, atol=0)
    actual = (actual / 2 + 0.5).clamp(0, 1)
    assert actual.shape == (1, 4, 32, 32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # A checkpoint naming its own schedule (Turbo's `model_index.sample_sigmas`, the first value
    # stored as an integer) runs that schedule, as upstream's pipeline configured with it does.
    schedule = [1, 0.8, 0.5]
    turbo_graph = build_qwen_image21(Config({
        **config, "processor": nested_config, "model_index": {"sample_sigmas": schedule}
    }))
    assert turbo_graph.processor_config == nested_config
    turbo_graph.components = graph.components
    turbo_model = QwenImage21Model.for_test(graph=turbo_graph, processor=processor)
    assert turbo_model.sample_sigmas == [1.0, 0.8, 0.5] and model.sample_sigmas is None
    embeds, mask = turbo_model.encode("a person in a blue jacket")
    latents = turbo_model.denoise(
        embeds,
        mask,
        width=32,
        height=32,
        steps=len(schedule),
        seed=44,
        on_step=lambda i: None,
        cancel=lambda: None,
    )
    distilled = (turbo_model.decode(latents, width=32, height=32) / 2 + 0.5).clamp(0, 1)
    upstream = QwenImage21Pipeline(
        **{**graph.components, "text_encoder": upstream_encoder},
        processor=processor,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(config["scheduler"]),
        sample_sigmas=[1.0, 0.8, 0.5],
    )
    expected = upstream(
        prompt="a person in a blue jacket",
        width=32,
        height=32,
        generator=torch.Generator().manual_seed(44),
        output_type="pt",
    ).images
    torch.testing.assert_close(distilled, expected, rtol=0, atol=0)
    assert not torch.equal(distilled, actual)

    assert {x.dtype for x in graph.components["vae"].parameters()} == {torch.float32}
    assert {x.dtype for x in graph.components["transformer"].parameters()} == {torch.bfloat16}
    shapes(model, tmp_path)
    # `generate_image_turbo` through the real kernel: the checkpoint's schedule wins over `steps`.
    turbo_run = invoke_with_fakes(
        app.get("generate_image_turbo"),
        {"prompt": "a person", "megapixels": 1, "steps": 5, "seed": 7},
        request_id="turbo",
        spool=tmp_path / "turbo",
        models={"model": turbo_model},
    )
    assert (turbo_run.result.width, turbo_run.result.height) == (1024, 1024)
    rows = [(row.kind, row.field, row.requested, row.applied) for row in turbo_run.adjustments]
    assert rows == [("clamp", "steps", 5, 3)], rows
    # The editing path: separately leased reference, prompt and denoise stages equal one upstream
    # call with the same reference. Upstream encodes references in the prompt's dtype, so both
    # run in float32 here.
    for module in (encoder, upstream_encoder, graph.components["transformer"]):
        module.float()
    reference = Image.linear_gradient("L").resize((96, 64)).convert("RGB")
    conditioning, reference_latents = model.encode_reference_images([reference])
    embeds, mask, pad_mask = model.encode_image_prompt("make it blue", conditioning)
    latents = model.denoise(
        embeds,
        mask,
        width=32,
        height=32,
        steps=2,
        seed=45,
        on_step=lambda i: None,
        cancel=lambda: None,
        reference_images=[reference],
        reference_latents=reference_latents,
        image_pad_mask=pad_mask,
    )
    edited = (model.decode(latents, width=32, height=32) / 2 + 0.5).clamp(0, 1)
    expected = pipe(
        prompt="make it blue",
        image=[reference],
        width=32,
        height=32,
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(45),
        output_type="pt",
    ).images
    torch.testing.assert_close(edited, expected, rtol=0, atol=0)


def progress() -> None:
    pipeline: Any = object.__new__(_StagedPipeline)
    pipeline.set_progress_bar_config(disable=True)
    for _ in range(2):
        frames: list[tuple[Any, ...]] = []
        with observing(lambda *frame, sink=frames: sink.append(frame)) as activity:
            with pipeline.progress_bar(total=2) as bar:
                bar.update(2)
        assert frames[-1][1:] == (False, True, True), (
            frames,
            [(type(t).__module__, type(t).__qualname__) for t in activity.unowned],
        )

    try:
        with pipeline.progress_bar(total=0):
            raise ValueError("author failure")
    except ValueError as exc:
        assert str(exc) == "author failure"
    else:
        raise AssertionError("progress swallowed author failure")

    frames = []
    with observing(lambda *frame: frames.append(frame)):
        assert list(pipeline.progress_bar(iterable=[1, 2])) == [1, 2]
        loop = event_loop()
        try:
            loop.run_until_complete(asyncio.sleep(0.01))
        finally:
            loop.close()
    assert all(frame[2] for frame in frames)
    assert any(not frame[1] and not frame[3] for frame in frames)

    release = threading.Event()
    raw = threading.Thread(target=release.wait)
    try:
        frames = []
        with observing(lambda *frame: frames.append(frame)):
            raw.start()
            with pipeline.progress_bar(total=1) as bar:
                bar.update()
        assert frames[-1][2] is False
    finally:
        release.set()
        raw.join()


if __name__ == "__main__":
    if sys.argv[1:] == ["progress"]:
        progress()
    else:
        with tempfile.TemporaryDirectory() as scratch:
            staged(Path(scratch))
        subprocess.run([sys.executable, __file__, "progress"], check=True)
        print("qwen-image-2 model: staged execution equals the Diffusers pipeline")
