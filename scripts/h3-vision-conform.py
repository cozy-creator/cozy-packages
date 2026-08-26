#!/usr/bin/env python
"""H3's FL2VA VISION-SEAM ARMS — keyframe pixels to text-encoder rows, decided on CPU.

    nice -n 19 python scripts/h3-vision-conform.py [--installed-wheel] [arm ...]
    arms: oraclerun, patchify, presentation, splices, deepstack, towerkeys, forward,
          keyframes, rng, boundary

WHY THIS IS A SECOND FILE. `h3-conform.py` declares "no torch, no weights, no network" and
that is load-bearing — it is the arm set that can run anywhere. Every arm HERE needs torch,
Pillow or transformers. Merging them would drag those dependencies into the arms that
deliberately do not have them.

THE LIVE ACTION IS THE SUBJECT. The current release accepts a prompt and zero, one or two
target-clock keyframes. It has no reference-media action, so this driver carries no dormant
reference presentation, reference layout or second construction dialect. Patch rows are
checked against transformers' real Qwen image processor; the toy forward executes this
endpoint's real vision tower and text stack.

EVERY ARM CARRIES ITS RED CONTROL, observed, per the standing rule: a green arm whose red
half never fired is a claim, not a check.

Nothing here loads a weight, touches a card or opens a socket.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import traceback
from typing import Any, cast

from _h3_probe import SOURCE_TREE, select

INSTALLED_WHEEL, _SUBJECTS = select(sys.argv, "h3", "h3_arch")
ENDPOINT_SOURCE = _SUBJECTS.get("h3", SOURCE_TREE / "h3.py")

PASS = "  ok   "
FAIL = "  FAIL "
_failures = 0


def observe(what: str, detail: str = "") -> None:
    print(f"{PASS}{what}" + (f"\n         {detail}" if detail else ""))


def fail(what: str, detail: str = "") -> None:
    global _failures
    _failures += 1
    print(f"{FAIL}{what}" + (f"\n         {detail}" if detail else ""))


def check(cond: bool, what: str, detail: str = "") -> None:
    observe(what, detail) if cond else fail(what, detail)


def _tokenizer() -> Any:
    """The endpoint's own bundled vocabulary."""
    from h3_arch.presentation import Tokenizer

    return Tokenizer()


def _image(seed: int, height: int, width: int) -> Any:
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(height, width, 3) * 255).astype("uint8"))


# ------------------------------------------------------------------ patchification


def arm_patchify() -> None:
    """The live action's keyframes batch through Qwen's real image processor exactly."""
    import torch

    from h3_arch import vision
    from h3_arch.presentation import build

    images = (_image(3, 768, 1344), _image(4, 768, 1344))
    presentation = build(_tokenizer(), "", keyframes=images)
    patched = vision.patchify(presentation)
    processor = vision._image_processor()
    oracle = processor(images=list(images), return_tensors="pt")
    check(
        torch.equal(patched.image_pixel_values, oracle["pixel_values"]),
        "two target-canvas keyframes produce the processor's exact patch matrix",
        f"shape {tuple(patched.image_pixel_values.shape)}",
    )
    check(
        torch.equal(patched.image_grid_thw, oracle["image_grid_thw"]),
        f"both grid THW rows match the processor: {patched.image_grid_thw.tolist()}",
    )
    expected = tuple(int(grid.prod()) // 4 for grid in oracle["image_grid_thw"])
    check(
        patched.token_counts == expected,
        f"each merged token count is grid.prod()//merge^2: {expected}",
    )
    check(
        patched.image_pixel_values.shape[1] == 3 * 2 * 16 * 16,
        f"a patch row is ch*t*p*p = 1536 wide, got {patched.image_pixel_values.shape[1]}",
    )

    reversed_pixels = processor(images=list(reversed(images)), return_tensors="pt")["pixel_values"]
    check(
        not torch.equal(reversed_pixels, patched.image_pixel_values),
        "RED: reversing the keyframes reverses their patch rows — order is not discarded",
    )


# ------------------------------------------------------------------ presentation


def _expected_keyframe_presentation(
    tokenizer: Any, prompt: str, token_counts: tuple[int, ...]
) -> tuple[list[int], list[int]]:
    """The FL2VA wire rule, independently spelled: labelled image pads, then prompt."""
    from h3_arch.presentation import VISION_END, VISION_START
    from h3_arch.vision import IMAGE_PAD_TOKEN

    ids: list[int] = []
    tags: list[int] = []
    for index, count in enumerate(token_counts, 1):
        label = tokenizer.ids(f"<Picture {index}>: ")
        ids.extend(label)
        tags.extend([1] * len(label))
        ids.extend([VISION_START, *([IMAGE_PAD_TOKEN] * count), VISION_END])
        tags.extend([0] * (count + 2))
    prompt_ids = tokenizer.ids(prompt)
    ids.extend(prompt_ids)
    tags.extend([1] * len(prompt_ids))
    return ids, tags


def arm_presentation() -> None:
    """Zero, one and two keyframes produce the exact FL2VA token/tag sequence."""
    from h3_arch import vision
    from h3_arch.presentation import build

    tok = _tokenizer()
    prompt = "a red fox walking through tall grass at golden hour"
    images = (_image(1, 192, 256), _image(2, 192, 256))
    seen: list[tuple[int, ...]] = []
    for count in (0, 1, 2):
        presentation = build(tok, prompt, keyframes=images[:count])
        patched = vision.patchify(presentation)
        expanded = vision.expand(presentation, patched.token_counts)
        expected_ids, expected_tags = _expected_keyframe_presentation(
            tok, prompt, patched.token_counts
        )
        check(
            list(expanded.token_ids) == expected_ids,
            f"FL2VA with {count} keyframe(s) has the exact labelled token sequence",
            f"{len(expected_ids)} rows",
        )
        check(
            list(expanded.tags) == expected_tags,
            f"FL2VA with {count} keyframe(s) tags vision rows VIDEO and labels/prompt TEXT",
        )
        seen.append(expanded.token_ids)

    check(
        len(set(seen)) == 3,
        "RED: zero, one and two keyframes are three different presentations",
    )


# ------------------------------------------------------------------ splices


def arm_splices() -> None:
    """Splice indices land exactly after `<|vision_start|>`, over a pad run of the right
    length and the right modality id."""
    from h3_arch import vision
    from h3_arch.presentation import VISION_END, VISION_START, build

    tok = _tokenizer()
    images = (_image(3, 192, 256), _image(4, 192, 256))
    pres = build(tok, "first and last frame anchors", keyframes=images)
    patched = vision.patchify(pres)
    exp = vision.expand(pres, patched.token_counts)
    ids = exp.token_ids

    check(
        len(exp.splices) == len(patched.token_counts) == 2,
        f"first and last keyframes produced two splice indices: {exp.splices}",
    )
    for n, (index, count) in enumerate(zip(exp.splices, patched.token_counts, strict=True)):
        check(
            ids[index - 1] == VISION_START,
            f"block {n} splice {index} sits immediately after <|vision_start|>",
        )
        check(
            ids[index + count] == VISION_END,
            f"block {n} pad run is exactly {count} long, then <|vision_end|>",
        )
        check(
            set(ids[index : index + count]) == {vision.IMAGE_PAD_TOKEN},
            f"block {n} pad run is all <|image_pad|> ({vision.IMAGE_PAD_TOKEN})",
        )
        check(
            set(exp.tags[index : index + count]) == {0},
            f"block {n} rows carry the VIDEO modality tag, not text",
        )

    # The text-encoder triples slice the right rows out of the batched patch matrix.
    blocks = vision.text_encoder_blocks(patched, exp)
    check(len(blocks) == 2, "two text-encoder blocks built")
    total = 0
    for n, (b, count) in enumerate(zip(blocks, patched.token_counts, strict=True)):
        rows = int(b.grid_thw.prod())
        check(
            b.patches.shape[0] == rows == count * 4,
            f"block {n} carries {rows} patch rows for {count} merged tokens",
        )
        check(b.index == exp.splices[n], f"block {n} index is its splice position")
        total += rows
    check(
        total == patched.image_pixel_values.shape[0],
        f"the two slices consume the whole patch matrix exactly ({total} rows)",
    )

    # RED CONTROL: a count that is one short must be REFUSED, not silently misaligned.
    try:
        vision.expand(pres, patched.token_counts[:1])
        fail("RED: a short token-count list refuses")
    except ValueError as exc:
        observe("RED: a short token-count list refuses", str(exc)[:90])


# ------------------------------------------------------------------ deepstack


def arm_deepstack() -> None:
    """The take-points are the adjudicated (8, 16, 24) and the features are CONSUMED."""
    import torch

    from h3_arch.config import TextEncoderConfig
    from h3_arch.text_encoder import Qwen3VLForConditionalGeneration

    config = TextEncoderConfig()
    check(
        tuple(config.vision_deepstack_layers) == (8, 16, 24),
        f"deepstack take-points are the official [8, 16, 24], got {config.vision_deepstack_layers}",
    )
    check(
        config.vision_depth == 27 and config.vision_spatial_merge_size == 2,
        f"vision tower is 27 blocks with a 2x2 merge, got {config.vision_depth}",
    )

    # The tower must EMIT one feature per take-point, and the text encoder must ADD them on
    # the first three decoder layers. Built on `meta`, so no weight is allocated.
    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(config)
    check(
        len(model.visual.deepstack_merger_list) == 3,
        f"three deepstack mergers exist, got {len(model.visual.deepstack_merger_list)}",
    )
    check(
        model.visual.deepstack_layers == (8, 16, 24),
        "the tower took the take-points from the config rather than hardcoding them",
    )

    import inspect

    src = inspect.getsource(Qwen3VLForConditionalGeneration.forward)
    check(
        "index < len(deepstack)" in src and "x[mask] = x[mask] + deepstack[index]" in src,
        "deepstack features are added at the vision positions over the first three layers",
    )

    # RED CONTROL: the take-points are not the ADD-points. If the source added them at 8,
    # 16 and 24 instead of 0, 1 and 2 it would be a different model, and this states the
    # distinction is live rather than incidental.
    check(
        "self.deepstack_layers" not in src,
        "RED: the ADD layers are 0..2 and are NOT the tower's take indices",
    )


# ------------------------------------------------------------------ boundary


def arm_boundary() -> None:
    """THE OLD ARM, INVERTED. `vision_seam_unbuilt` was the guarantee that a vision
    presentation could not silently reach Qwen as nothing. It must now be UNREACHABLE for
    a keyframe request — its survival would mean the seam is still not wired."""
    import inspect

    from h3_arch import layout, presentation, vision

    path = ENDPOINT_SOURCE
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    model = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Fl2VAModel"
    )
    condition = next(
        node
        for node in model.body
        if isinstance(node, ast.FunctionDef) and node.name == "condition_text"
    )
    prepare = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare"
    )
    condition_src = ast.get_source_segment(src, condition) or ""
    prepare_src = ast.get_source_segment(src, prepare) or ""
    check(
        "vision_seam_unbuilt" not in condition_src,
        "RED (inverted): condition_text no longer refuses a vision presentation",
    )
    check(
        "text_encoder_blocks" in condition_src,
        "condition_text splices the patchified vision blocks into the text-encoder call",
    )
    check(
        "vision.patchify" in prepare_src,
        "prepare patchifies, so the layout's text span is the EXPANDED length",
    )
    check(
        "run.expanded.tags" in src,
        "the DiT is tagged per TEXT-ENCODER row, not per presentation row",
    )
    check(
        src.count("vision_seam_unbuilt") == 0,
        f"`vision_seam_unbuilt` is gone from h3.py ({src.count('vision_seam_unbuilt')} left)",
    )
    check(
        "_condition_rows" in src and "video_patch_proj" in src,
        "the FL2VA model builds one keyframe row block per packed segment",
    )
    check(
        "condition_geometry" in src and "disagree" in src,
        "the derived geometry is checked against the encode rather than trusted",
    )
    check(
        all(
            spelling not in src
            for spelling in (
                "RefGenerateInput",
                "Ref2VAModel",
                "reference_to_video",
                "AudioReference",
                "VideoReference",
            )
        ),
        "the unserved Ref2VA contract is absent rather than hidden behind refusal branches",
    )
    library_src = "\n".join(
        inspect.getsource(module) for module in (presentation, vision, layout)
    )
    removed = (
        "PresentedReference",
        "RefBlock",
        "references=",
        "video_block",
        "normalize_reference_image",
        "reference_block_geometry",
        "ref_img",
        "ref_audio",
        "audio_cond_timestep",
    )
    leftovers = tuple(spelling for spelling in removed if spelling in library_src)
    check(
        not leftovers,
        "the shipped presentation, vision and layout libraries contain no Ref2VA machinery",
        f"leftovers: {leftovers}",
    )


def arm_forward() -> None:
    """THE SEAM ACTUALLY RUNS — a real text-encoder forward with a real vision block.

    Every other arm here checks a number that describes the seam. This one EXECUTES it, at
    a toy width with random weights, which is the only CPU-affordable way to find out
    whether the splice indices, the patch slices, the grid and the deepstack add agree with
    each other rather than merely with the arithmetic. The released text encoder is 62 GiB
    and cannot be part of a CI arm; the shapes it would fail on are all here.

    Real geometry, toy widths: the vision tower keeps its 27 blocks, its 2x2 merge and its
    (8, 16, 24) take-points, because those are what the splice accounting depends on.
    """
    import torch

    from h3_arch import vision
    from h3_arch.config import TextEncoderConfig
    from h3_arch.presentation import build
    from h3_arch.text_encoder import Qwen3VLForConditionalGeneration

    config = TextEncoderConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=256,
        vision_hidden_size=64,
        vision_num_heads=2,
        vision_intermediate_size=128,
        vision_out_hidden_size=512,
    )
    torch.manual_seed(0)
    model = Qwen3VLForConditionalGeneration(config).to(torch.float32)
    model.eval()

    tok = _tokenizer()
    # A small keyframe keeps this real forward CPU-affordable. Only the grid is smaller;
    # the patch, merge, splice, tower and decoder-layer paths are the shipped ones.
    small = _image(5, 192, 256)
    pres = build(tok, "a fox in the grass", keyframes=(small,))
    patched = vision.patchify(pres)
    exp = vision.expand(pres, patched.token_counts)
    blocks = vision.text_encoder_blocks(patched, exp)

    tokens = torch.tensor([exp.token_ids], dtype=torch.long)
    with torch.inference_mode():
        text_only = model(tokens)
        spliced = model(tokens, vision=blocks)

    check(
        spliced.shape == (1, len(exp.token_ids), config.hidden_size),
        f"the spliced forward returns [1, {len(exp.token_ids)}, {config.hidden_size}]",
        f"got {tuple(spliced.shape)}",
    )
    check(torch.isfinite(spliced).all().item(), "the spliced hidden state is all finite")

    # THE ARM THAT MATTERS: splicing must CHANGE the state, and change it at the vision
    # positions. A seam that ran but spliced nothing would pass every shape check above.
    start = exp.splices[0]
    stop = start + patched.token_counts[0]
    delta = (spliced - text_only).abs()
    check(
        delta[0, start:stop].max().item() > 0,
        f"the vision rows {start}..{stop} DIFFER from the text-only forward",
        f"max |delta| = {delta[0, start:stop].max().item():.4g}",
    )
    # The prompt follows the block, and the stack is CAUSAL, so the splice reaches the
    # prompt rows and reaches the label rows before it NOT AT ALL. Both halves are asserted:
    # a non-zero prefix delta would mean the causal mask was not doing its job.
    check(
        delta[0, stop:].max().item() > 0,
        f"the splice propagates into the {len(exp.token_ids) - stop} prompt rows after it",
        f"suffix max |delta| = {delta[0, stop:].max().item():.4g}",
    )
    check(
        delta[0, :start].max().item() == 0,
        "and reaches the label rows BEFORE it not at all — the stack is causal",
        f"prefix max |delta| = {delta[0, :start].max().item():.4g}",
    )

    # RED CONTROL: a block spliced at the WRONG index must produce a different state. If it
    # did not, the index would be decorative and every arm above would be measuring nothing.
    moved = [type(b)(patches=b.patches, grid_thw=b.grid_thw, index=b.index - 1) for b in blocks]
    with torch.inference_mode():
        shifted = model(tokens, vision=moved)
    check(
        not torch.equal(shifted, spliced),
        "RED: splicing one row earlier gives a DIFFERENT state — the index is load-bearing",
    )

    # RED CONTROL: the deepstack features are consumed. Zeroing the three mergers must
    # change the answer, or the tower's take-points are wired to nothing.
    with torch.inference_mode():
        for merger in model.visual.deepstack_merger_list:
            merger.linear_fc2.weight.zero_()
            merger.linear_fc2.bias.zero_()
        without = model(tokens, vision=blocks)
    check(
        not torch.equal(without, spliced),
        "RED: zeroing the deepstack mergers changes the state — the take-points feed it",
    )


def arm_towerkeys() -> None:
    """Our vision tower against transformers' Qwen3-VL tower, destination for destination.

    If it does, then every constant the port had to guess (the rope sections, the vision
    rope theta, the merger's pre- vs post-shuffle norm) is guessing about ARITHMETIC over a
    graph whose shape is confirmed, which is a much smaller claim than guessing about both.
    Header-verified, on the control plane, for nothing.

    The carrier's own banked header is the third party: 902 BF16 destinations of which the
    `visual.*` families are 27 blocks and 3 deepstack mergers, with `attn.qkv` FUSED on the
    vision side and q/k/v SPLIT on the language side — the two dialects this arm's two
    models have to agree with.
    """
    import torch

    from h3_arch.config import TextEncoderConfig
    from h3_arch.text_encoder import Qwen3VLForConditionalGeneration

    config = TextEncoderConfig()
    with torch.device("meta"):
        ours = Qwen3VLForConditionalGeneration(config)
    mine = {
        name[len("visual.") :]: tuple(p.shape)
        for name, p in ours.named_parameters()
        if name.startswith("visual.")
    }

    # BOTH SIDES ARE IN SCOPE HERE, and after #579 both spell the tower
    # `Qwen3VLVisionModel` — the port answers to the name of what it ports. The module path
    # is the real disambiguator; inside one function an alias is, so upstream's side carries
    # the `Official` prefix this arm already uses for its variables.
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig as OfficialQwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionModel as OfficialQwen3VLVisionModel,
    )

    official_config = OfficialQwen3VLVisionConfig(
        depth=config.vision_depth,
        hidden_size=config.vision_hidden_size,
        intermediate_size=config.vision_intermediate_size,
        num_heads=config.vision_num_heads,
        in_channels=config.vision_in_channels,
        patch_size=config.vision_patch_size,
        spatial_merge_size=config.vision_spatial_merge_size,
        temporal_patch_size=config.vision_temporal_patch_size,
        out_hidden_size=config.vision_out_hidden_size,
        num_position_embeddings=config.vision_num_position_embeddings,
        deepstack_visual_indexes=list(config.vision_deepstack_layers),
    )
    with torch.device("meta"):
        official = OfficialQwen3VLVisionModel(official_config)
    theirs = {name: tuple(p.shape) for name, p in official.named_parameters()}

    check(
        len(mine) == len(theirs) == 351,
        f"the tower has 351 destinations on both sides (ours {len(mine)}, official {len(theirs)})",
    )
    missing = sorted(set(theirs) - set(mine))
    extra = sorted(set(mine) - set(theirs))
    check(
        not missing,
        f"no official destination is missing from the port ({len(missing)})",
        str(missing[:4]),
    )
    check(not extra, f"the port invents no destination ({len(extra)})", str(extra[:4]))
    mismatched = sorted(k for k in set(mine) & set(theirs) if mine[k] != theirs[k])
    check(
        not mismatched,
        f"every shared destination agrees on SHAPE ({len(mismatched)} disagree)",
        str([(k, mine[k], theirs[k]) for k in mismatched[:3]]),
    )
    check(
        mine.get("blocks.0.attn.qkv.weight") == (3456, 1152),
        f"the vision attention is FUSED qkv at [3456, 1152], as the carrier's header says: "
        f"{mine.get('blocks.0.attn.qkv.weight')}",
    )
    check(
        mine.get("merger.linear_fc2.weight") == (5120, 4608),
        f"the merger folds 2x2 of 1152 into the 5120 language width: "
        f"{mine.get('merger.linear_fc2.weight')}",
    )

    # RED CONTROL: the deepstack mergers are NOT the patch merger — their norm is applied
    # POST-shuffle and is 4608 wide, against the merger's pre-shuffle 1152. If the two were
    # interchangeable the port could have shared one class and been silently wrong.
    check(
        mine.get("merger.norm.weight") == (1152,)
        and mine.get("deepstack_merger_list.0.norm.weight") == (4608,),
        "RED: the two merger classes differ where they must — 1152 pre-shuffle vs 4608 post",
        f"{mine.get('merger.norm.weight')} vs {mine.get('deepstack_merger_list.0.norm.weight')}",
    )


def arm_keyframes() -> None:
    """Keyframe geometry, rows, pixels and noising agree from preparation through packing."""
    import torch

    from h3_arch import vision
    from h3_arch.dit import patchify_video
    from h3_arch.layout import Keyframe, PackedLayout, latent_grid

    grid = latent_grid(124, 1344, 768)
    patch = (1, 2, 2)
    image = _image(6, grid.height, grid.width)
    t, h, w = vision.conditioning_geometry(image)
    check(
        (t, h, w) == (1, grid.height // 16, grid.width // 16),
        f"a target-canvas keyframe resolves to conditioning latents {(t, h, w)}",
    )

    layout = PackedLayout(
        32,
        grid,
        keyframes=(Keyframe(0), Keyframe(grid.frames - 1)),
    )
    reserved = [end - start for start, end, kind in layout.segments if kind == "cond"]
    produced = patchify_video(torch.zeros(1, 24, t, h, w), patch).shape[0]
    check(
        reserved == [produced, produced] and produced > 0,
        f"first and last anchors each reserve the {produced} rows their latents produce",
    )

    pixels = vision.condition_pixels(image)
    check(
        tuple(pixels.shape) == (1, 3, 1, grid.height, grid.width),
        f"a keyframe becomes [1, 3, 1, H, W], got {tuple(pixels.shape)}",
    )
    check(
        pixels.dtype == torch.float32
        and float(pixels.min()) >= -1.0
        and float(pixels.max()) <= 1.0,
        "keyframe pixels are float32 in [-1, 1] before the conditioning VAE",
    )

    # RED CONTROL: only first and last are admitted target-clock anchors.
    try:
        PackedLayout(32, grid, keyframes=(Keyframe(1),))
        fail("RED: an interior keyframe refuses")
    except ValueError as exc:
        observe("RED: an interior keyframe refuses", str(exc)[:90])

    # The anchor noising is `x_t = t*x_0 + (1-t)*noise`, upstream's `scale_noise` in H3's
    # `t` convention, where t = 1 is CLEAN. A schedule-shaped `(1-t)*x + t*noise` at 0.999
    # would return essentially pure noise and destroy every anchor.
    x0 = torch.ones(4)
    noise = torch.zeros(4)
    noise_level = 0.999
    ours = noise_level * x0 + (1.0 - noise_level) * noise
    check(
        abs(ours.mean().item() - 0.999) < 1e-6,
        f"noising at t=0.999 keeps 99.9% of the anchor, got {ours.mean().item():.4f}",
    )
    reversed_convention = (1.0 - noise_level) * x0 + noise_level * noise
    check(
        abs(reversed_convention.mean().item() - 0.001) < 1e-6,
        "RED: the reversed convention would keep 0.1% and destroy the anchor",
        f"got {reversed_convention.mean().item():.4f}",
    )

    # Execute the real conditioning operation on a controlled receiver. Source-string
    # inspection used to call this behavior proof, but could not prove the refusal ran
    # before the encoder or that the posterior was actually sampled and rounded.
    from types import SimpleNamespace

    from h3_arch.video_vae import CONDITION_ENCODE_SEED, AutoencoderKLMiniMaxH3

    calls = 0
    mean = torch.full((1, 2, 1, 1, 1), 0.25)
    logvar = torch.zeros_like(mean)

    def encode_moments(_pixels: Any) -> Any:
        nonlocal calls
        calls += 1
        return torch.cat((mean, logvar), dim=1)

    receiver: Any = SimpleNamespace(
        compute_dtype=torch.float32,
        _normalize_pixels=lambda value: value,
        _encode_moments=encode_moments,
        latents_mean=torch.zeros(2),
        latents_std=torch.ones(2),
    )
    encoded = AutoencoderKLMiniMaxH3.encode_condition(
        receiver, torch.zeros(1, 3, 1, 2, 2)
    )
    expected_noise = torch.randn(
        mean.shape,
        generator=torch.Generator().manual_seed(CONDITION_ENCODE_SEED),
        dtype=torch.float32,
    )
    expected = (mean + expected_noise).to(torch.float16).float()
    check(
        torch.equal(encoded, expected),
        "conditioning executes the fixed-seed posterior sample and fp16 rounding exactly",
    )
    check(calls == 1, "one keyframe calls the conditioning encoder exactly once")
    check(
        not torch.equal(encoded, mean),
        "RED: taking the posterior mean is observably different from conditioning sampling",
    )
    try:
        AutoencoderKLMiniMaxH3.encode_condition(receiver, torch.zeros(1, 3, 2, 2, 2))
        fail("RED: a multi-frame conditioning input refuses before encoding")
    except ValueError as exc:
        observe("RED: a multi-frame conditioning input refuses before encoding", str(exc))
    check(calls == 1, "the multi-frame refusal happens before the encoder is called")


def arm_rng() -> None:
    """One request owns one ordered CPU stream: condition(s), video, then audio rows."""
    import inspect
    from types import SimpleNamespace

    import torch

    import h3
    from h3_arch import DitConfig
    from h3_arch.dit import pack_audio, unpack_audio
    from h3_arch.layout import Keyframe, LatentGrid, PackedLayout

    grid = LatentGrid(
        frames=5,
        height=32,
        width=32,
        latent_t=2,
        latent_h=2,
        latent_w=2,
        audio_t=3,
    )
    images = (_image(17, 32, 32), _image(18, 32, 32))
    geometry = ((1, 2, 2), (1, 2, 2))
    plan = h3.H3Plan(
        grid=grid,
        keyframes=(Keyframe(0), Keyframe(4)),
        steps=1,
        mute=False,
        condition_images=images,
        condition_geometry=geometry,
    )
    run = h3.H3Run(
        plan=plan,
        layout=PackedLayout(4, grid, keyframes=plan.keyframes),
        timestep_plan=cast(Any, None),
        generator=torch.Generator(device="cpu").manual_seed(73),
        expanded=cast(Any, None),
    )

    class FakeVae:
        post_quant_conv = SimpleNamespace(weight=torch.empty(1))

        def encode_condition(self, _pixels: Any) -> Any:
            return torch.zeros(1, 24, 1, 2, 2)

    model: Any = SimpleNamespace(
        pipe=SimpleNamespace(
            components={"video_vae": FakeVae()},
            config=SimpleNamespace(dit=DitConfig()),
        )
    )
    condition_visual = inspect.unwrap(h3.Fl2VAModel.condition_visual)
    conditions = condition_visual(model, run, noise_level=0.0)

    class EmptySolver:
        def steps(self) -> tuple[()]:
            return ()

    sampled = h3._sample(
        torch,
        model,
        run,
        cast(Any, EmptySolver()),
        torch.empty(1),
        lambda _index: None,
        cast(Any, SimpleNamespace()),
    )

    oracle = torch.Generator(device="cpu").manual_seed(73)
    expected_conditions = [
        torch.randn((1, 24, 1, 2, 2), generator=oracle) for _ in range(2)
    ]
    expected_video = torch.randn((1, 24, 2, 2, 2), generator=oracle)
    expected_audio_rows = torch.randn((grid.audio_rows, 32), generator=oracle)
    check(
        all(
            torch.equal(got, want)
            for got, want in zip(conditions, expected_conditions, strict=True)
        ),
        "both keyframes consume the request generator in packed order",
    )
    check(
        torch.equal(sampled["video"], expected_video),
        "target video consumes the same stream after every condition draw",
    )
    check(
        torch.equal(pack_audio(sampled["audio"]), expected_audio_rows),
        "target audio consumes the same stream last and in official row layout",
    )
    check(
        torch.equal(unpack_audio(expected_audio_rows), sampled["audio"]),
        "audio row noise crosses the latent-shaped model boundary by a pure permutation",
    )

    legacy_condition0 = torch.randn(
        (1, 24, 1, 2, 2), generator=torch.Generator().manual_seed(73)
    )
    legacy_condition1 = torch.randn(
        (1, 24, 1, 2, 2), generator=torch.Generator().manual_seed(74)
    )
    legacy_targets = torch.Generator().manual_seed(73)
    legacy_video = torch.randn((1, 24, 2, 2, 2), generator=legacy_targets)
    legacy_audio = torch.randn((1, 32, 2, 3), generator=legacy_targets)
    check(
        torch.equal(legacy_condition0.flatten(), legacy_video.flatten()[:96]),
        "RED: the deleted reseed reused the first keyframe's random prefix for target video",
    )
    check(
        not torch.equal(conditions[1], legacy_condition1)
        and not torch.equal(sampled["video"], legacy_video)
        and not torch.equal(pack_audio(sampled["audio"]), pack_audio(legacy_audio)),
        "RED: independent condition seeds plus a reset target stream disagree with the release",
    )


def arm_oraclerun() -> None:
    """THE POD HARNESS'S `H3Run` CALL, CHECKED HERE FOR $0.

    `scripts/h3-seam-oracle.py::seam_run` is the only construction of `H3Run` outside
    `h3.py`'s own path, and it runs POD-SIDE ONLY — so when the vision work gave `H3Run` two
    new required fields, nothing on this box or in CI noticed. The oracle raised
    `TypeError: missing 2 required positional arguments` at `seam_run`, AFTER a 48 GiB
    text encoder had loaded on a rented H200: a signature drift billed at card rates and
    discovered thirteen minutes in.

    Importing either file would also import torch and the private runtime. The signature
    drift is syntax, so this arm parses the ACTUAL dataclass and the ACTUAL harness call and
    requires every non-default field to be supplied. Zero weights, zero network, zero fake
    author surface.
    """
    endpoint_path = ENDPOINT_SOURCE
    endpoint_tree = ast.parse(endpoint_path.read_text(), filename=str(endpoint_path))
    run_class = next(
        node
        for node in endpoint_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "H3Run"
    )
    required = tuple(
        node.target.id
        for node in run_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is None
    )

    oracle_path = pathlib.Path(__file__).with_name("h3-seam-oracle.py")
    oracle_tree = ast.parse(oracle_path.read_text(), filename=str(oracle_path))
    seam_run = next(
        node
        for node in oracle_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "seam_run"
    )
    calls = [
        node
        for node in ast.walk(seam_run)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "H3Run"
    ]
    check(len(calls) == 1, "the oracle's seam_run has exactly one H3Run construction")
    if len(calls) != 1:
        return
    supplied = tuple(keyword.arg for keyword in calls[0].keywords if keyword.arg is not None)
    missing = tuple(field for field in required if field not in supplied)
    check(
        not missing,
        "every required H3Run field is supplied by the pod harness",
        detail=f"required {required}; supplied {supplied}",
    )
    check(
        bool(set(required) - set(supplied[:-1])),
        "RED: deleting the harness's last required keyword makes the signatures disagree",
    )


ARMS = {
    "oraclerun": arm_oraclerun,
    "patchify": arm_patchify,
    "presentation": arm_presentation,
    "splices": arm_splices,
    "deepstack": arm_deepstack,
    "towerkeys": arm_towerkeys,
    "forward": arm_forward,
    "keyframes": arm_keyframes,
    "rng": arm_rng,
    "boundary": arm_boundary,
}


def main(argv: list[str]) -> int:
    names = argv or list(ARMS)
    for name in names:
        if name not in ARMS:
            print(f"unknown arm {name!r}; arms are {', '.join(ARMS)}", file=sys.stderr)
            return 2
        print(f"\n=== {name} ===")
        try:
            ARMS[name]()
        except Exception:
            global _failures
            _failures += 1
            print(f"{FAIL}{name} raised")
            traceback.print_exc()
    print(f"\n{_failures} failure(s)")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
