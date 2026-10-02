#!/usr/bin/env python
"""Real tiny Qwen modules: the package's staged model agrees exactly with Diffusers' pipeline.

Run with the package's own interpreter. The two parts run in separate processes: an executor
owns a fresh Python process, and its activity observer is process state.
"""

from __future__ import annotations

import asyncio
import base64
import subprocess
import sys
import tempfile
import threading
import zlib
from pathlib import Path
from typing import Any

import torch
from cozy_runtime.author import Config
from cozy_runtime.author._activity import event_loop, observing
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "qwen-image-2"))

from qwen_image_2.model import (  # noqa: E402
    QwenImage21Model,
    _StagedPipeline,
    build_processor,
    build_qwen_image21,
)


def staged(tmp_path: Path) -> None:
    config = {
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
            patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=1024, max_pixels=4096
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, merge_size=2, temporal_patch_size=2),
        tokenizer=tok,
        chat_template=chat,
    )
    processor_bundle = {
        "tokenizer": base64.urlsafe_b64encode(
            zlib.compress(tok.backend_tokenizer.to_str().encode())
        ).decode(),
        "tokenizer_config": {"unk_token": "[UNK]", "pad_token": "[PAD]"},
        "image_processor": {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2},
        "video_processor": {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2},
        "chat_template": chat,
    }
    rebuilt = build_processor(processor_bundle)
    assert rebuilt.tokenizer.encode("blue jacket") == processor.tokenizer.encode("blue jacket")
    extended = build_processor(dict(processor_bundle, added_by_a_newer_ingester=True))
    assert extended.tokenizer.encode("blue jacket") == processor.tokenizer.encode("blue jacket")
    for invalid in (
        {key: value for key, value in processor_bundle.items() if key != "chat_template"},
        dict(processor_bundle, tokenizer="invalid"),
        dict(
            processor_bundle,
            tokenizer=base64.urlsafe_b64encode(zlib.compress(b"x") + b"extra").decode(),
        ),
    ):
        try:
            build_processor(invalid)
        except ValueError:
            continue
        raise AssertionError("an invalid processor bundle was accepted")
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

    assert {x.dtype for x in graph.components["vae"].parameters()} == {torch.float32}
    assert {x.dtype for x in graph.components["transformer"].parameters()} == {torch.bfloat16}


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
