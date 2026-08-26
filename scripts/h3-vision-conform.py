#!/usr/bin/env python
"""H3's VISION-SEAM CONFORMANCE ARMS — the pixels-to-text-encoder seam, decided on CPU.

    nice -n 19 python scripts/h3-vision-conform.py [--installed-wheel] [arm ...]
    arms: oraclerun, preprocess, presentation, splices, deepstack, towerkeys, forward,
          refrows, refusal

WHY THIS IS A SECOND FILE. `h3-conform.py` declares "no torch, no weights, no network" and
that is load-bearing — it is the arm set that can run anywhere. Every arm HERE needs torch,
transformers and the pinned diffusers, because its oracle IS upstream: the same processor
and the same presentation builder the official implementation runs. Merging them would drag
a torch dependency into the arms that deliberately do not have one.

THE ORACLE IS CALLED, NOT PARAPHRASED (#531). `h3-conform.py` deliberately re-spells its
reference expressions from upstream SOURCE so the arm does not inherit the thing it checks.
That is right for a closed-form scalar and wrong for a 1,536-wide patch matrix: a
transcribed patchifier is a second implementation with its own bugs, and the question here
is whether OUR seam agrees with UPSTREAM'S, which is answered by running both. So these arms
import `diffusers.modular_pipelines.minimax_h3` and `transformers` and diff against them.
`MiniMaxH3Ref2VATextEncoderStep._build_presentation` is a `@staticmethod` taking a tokenizer
and plain lists, which makes it callable as a pure oracle with no pipeline and no weights.

EVERY ARM CARRIES ITS RED CONTROL, observed, per the standing rule: a green arm whose red
half never fired is a claim, not a check.

Nothing here loads a weight, touches a card or opens a socket.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import traceback
from typing import Any

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
    """The endpoint's OWN bundled vocabulary, handed to both sides of every arm.

    Using one tokenizer for the oracle and the subject is deliberate: these arms decide
    whether the ASSEMBLY agrees, and a slow/fast tokenizer delta would show up as an
    assembly disagreement it is not.
    """
    from h3_arch.presentation import Tokenizer

    return Tokenizer()


def _image(seed: int, height: int, width: int) -> Any:
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(height, width, 3) * 255).astype("uint8"))


# ------------------------------------------------------------------ preprocess


def arm_preprocess() -> None:
    """Our normalization + patchify must equal upstream's, byte for byte.

    Upstream's rule lives in `MiniMaxH3Ref2VASetupStep.__call__` (short edge 2048, both
    axes rounded onto 32, aspect refused outside 1:4..4:1) and then hands the result to
    `Qwen3VLProcessor.image_processor`. Ours is `vision.normalize_reference_image` and
    `vision.patchify`. The arm runs both and diffs the tensors.
    """
    import torch
    from PIL import Image

    from h3_arch import vision

    # Upstream's normalization, re-run here through the SAME primitives the pipeline uses,
    # so the comparison is against upstream's arithmetic rather than against a restatement.
    def upstream_normalize(image: Any) -> Any:
        multiple, short_edge = 32, 2048
        width, height = image.size
        scale = short_edge / min(width, height)
        target_height = max(multiple, round(height * scale / multiple) * multiple)
        target_width = max(multiple, round(width * scale / multiple) * multiple)
        if image.size != (target_width, target_height):
            image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return image

    for seed, (h, w) in enumerate([(512, 768), (900, 900), (480, 1800), (2000, 1000)]):
        raw = _image(seed, h, w)
        ours = vision.normalize_reference_image(raw)
        theirs = upstream_normalize(raw)
        check(
            ours.size == theirs.size,
            f"reference canvas {w}x{h} -> {ours.size[0]}x{ours.size[1]}",
            f"upstream {theirs.size[0]}x{theirs.size[1]}",
        )
        import numpy as np

        check(
            np.array_equal(np.asarray(ours), np.asarray(theirs)),
            f"reference pixels {w}x{h} are byte-identical to upstream's LANCZOS",
        )
        check(
            ours.size[0] % 32 == 0 and ours.size[1] % 32 == 0,
            f"reference canvas {ours.size} lands on the 32 grid",
        )
        check(
            min(ours.size) == 2048 or max(ours.size) >= 2048,
            f"reference short edge is the checkpoint's 2048, got {min(ours.size)}",
        )

    # The aspect band REFUSES rather than letter-boxing.
    try:
        vision.normalize_reference_image(_image(9, 100, 900))
        fail("a 9:1 reference image refuses")
    except ValueError as exc:
        observe("a 9:1 reference image refuses", str(exc)[:90])

    # Patchify against the processor directly.
    from h3_arch.presentation import Presentation, VisionBlock

    img = vision.normalize_reference_image(_image(3, 512, 768))
    pres = Presentation((151652, VisionBlock(img, False), 151653), (0, 0, 0))
    patched = vision.patchify(pres)
    processor = vision._image_processor()
    oracle = processor(images=[img], return_tensors="pt")
    check(
        torch.equal(patched.image_pixel_values, oracle["pixel_values"]),
        "patch matrix is bit-identical to the official processor's",
        f"shape {tuple(patched.image_pixel_values.shape)}",
    )
    check(
        torch.equal(patched.image_grid_thw, oracle["image_grid_thw"]),
        f"grid THW matches the official processor's: {patched.image_grid_thw.tolist()}",
    )
    expected = int(oracle["image_grid_thw"][0].prod()) // 4
    check(
        patched.token_counts == (expected,),
        f"merged token count is grid.prod()//merge^2 = {expected}",
    )
    check(
        patched.image_pixel_values.shape[1] == 3 * 2 * 16 * 16,
        f"a patch row is ch*t*p*p = 1536 wide, got {patched.image_pixel_values.shape[1]}",
    )

    # RED CONTROL: the geometry the seam would have had if the short edge were the target
    # canvas's 768 rather than the reference rule's 2048. It must DISAGREE.
    wrong = img.resize((768, 512), Image.Resampling.LANCZOS)
    wrong_grid = processor(images=[wrong], return_tensors="pt")["image_grid_thw"]
    check(
        not torch.equal(wrong_grid, patched.image_grid_thw),
        "RED: the 768-canvas rule disagrees with the 2048 reference rule",
        f"{wrong_grid.tolist()} vs {patched.image_grid_thw.tolist()}",
    )


# ------------------------------------------------------------------ presentation


def _upstream_presentation(tokenizer: Any, prompt: str, refs: list[Any], counts: list[int]) -> Any:
    """Upstream's own `_build_presentation`, called as a pure function."""
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    class _Tok:
        """Adapts the bundled tokenizer to the two calls upstream makes on it."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
            return {"input_ids": self._inner.ids(text)}

        def convert_tokens_to_ids(self, token: str) -> int:
            return {
                "<|vision_start|>": 151652,
                "<|vision_end|>": 151653,
                "<|image_pad|>": 151655,
                "<|video_pad|>": 151656,
            }[token]

    return MiniMaxH3Ref2VATextEncoderStep._build_presentation(
        _Tok(tokenizer), prompt, refs, counts, [], [], text_tag=1, video_tag=0
    )


def arm_presentation() -> None:
    """Our expanded token ids must EQUAL upstream's, element for element."""
    from h3_arch import vision
    from h3_arch.presentation import PresentedReference, build

    tok = _tokenizer()
    prompt = "a red fox walking through tall grass at golden hour"

    class _Ref:
        def __init__(self, kind: str, has_audio: bool = False) -> None:
            self.kind, self.has_audio = kind, has_audio

    # One image reference.
    images = [vision.normalize_reference_image(_image(1, 512, 768))]
    pres = build(tok, prompt, references=(PresentedReference(kind="image", pixels=images[0]),))
    patched = vision.patchify(pres)
    ours = vision.expand(pres, patched.token_counts)
    theirs_ids, theirs_tags = _upstream_presentation(
        tok, prompt, [_Ref("image")], list(patched.token_counts)
    )
    check(
        list(ours.token_ids) == theirs_ids,
        f"ref2va(1 image) token ids equal upstream's, {len(theirs_ids)} tokens",
    )
    check(list(ours.tags) == theirs_tags, "ref2va(1 image) modality tags equal upstream's")

    # Three image references — the labels must number 1,2,3 in request order.
    imgs = [vision.normalize_reference_image(_image(i, 480 + 40 * i, 640)) for i in range(3)]
    pres3 = build(
        tok,
        prompt,
        references=tuple(PresentedReference(kind="image", pixels=i) for i in imgs),
    )
    patched3 = vision.patchify(pres3)
    ours3 = vision.expand(pres3, patched3.token_counts)
    theirs3, tags3 = _upstream_presentation(
        tok, prompt, [_Ref("image")] * 3, list(patched3.token_counts)
    )
    check(
        list(ours3.token_ids) == theirs3,
        f"ref2va(3 images) token ids equal upstream's, {len(theirs3)} tokens",
    )
    check(list(ours3.tags) == tags3, "ref2va(3 images) modality tags equal upstream's")

    # An audio reference alongside images: the label order is the arm.
    pres_a = build(
        tok,
        prompt,
        references=(
            PresentedReference(kind="image", pixels=imgs[0]),
            PresentedReference(kind="audio", has_audio=True),
        ),
    )
    patched_a = vision.patchify(pres_a)
    ours_a = vision.expand(pres_a, patched_a.token_counts)
    theirs_a, _ = _upstream_presentation(
        tok, prompt, [_Ref("image"), _Ref("audio", True)], list(patched_a.token_counts)
    )
    check(list(ours_a.token_ids) == theirs_a, "ref2va(image + audio) token ids equal upstream's")

    # RED CONTROL, and the defect it is a control for is REAL: keying the audio label on
    # `kind == "audio"` instead of on `has_audio` drops the label of every SOUNDTRACKED
    # VIDEO, which silently renumbers every later audio reference in the same request.
    with_sound = PresentedReference(kind="image", pixels=imgs[0], has_audio=True)
    pres_s = build(tok, prompt, references=(with_sound,))
    ours_s = vision.expand(pres_s, vision.patchify(pres_s).token_counts)
    pres_n = build(tok, prompt, references=(PresentedReference(kind="image", pixels=imgs[0]),))
    ours_n = vision.expand(pres_n, vision.patchify(pres_n).token_counts)
    check(
        list(ours_s.token_ids) != list(ours_n.token_ids),
        "RED: a sound-bearing reference presents DIFFERENTLY from a silent one",
        f"{len(ours_s.token_ids)} vs {len(ours_n.token_ids)} tokens",
    )
    theirs_s, _ = _upstream_presentation(
        tok, prompt, [_Ref("image", True)], list(vision.patchify(pres_s).token_counts)
    )
    check(
        list(ours_s.token_ids) == theirs_s,
        "a sound-bearing image reference matches upstream's <Audio 1>: <Picture 1>: order",
    )


# ------------------------------------------------------------------ splices


def arm_splices() -> None:
    """Splice indices land exactly after `<|vision_start|>`, over a pad run of the right
    length and the right modality id."""
    from h3_arch import vision
    from h3_arch.presentation import VISION_END, VISION_START, PresentedReference, build

    tok = _tokenizer()
    imgs = [vision.normalize_reference_image(_image(i, 512, 512 + 32 * i)) for i in range(3)]
    pres = build(
        tok,
        "three references",
        references=tuple(PresentedReference(kind="image", pixels=i) for i in imgs),
    )
    patched = vision.patchify(pres)
    exp = vision.expand(pres, patched.token_counts)
    ids = exp.token_ids

    check(
        len(exp.splices) == len(patched.token_counts) == 3,
        f"three blocks produced three splice indices: {exp.splices}",
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
    check(len(blocks) == 3, "three text-encoder blocks built")
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
        f"the three slices consume the whole patch matrix exactly ({total} rows)",
    )

    # RED CONTROL: a count that is one short must be REFUSED, not silently misaligned.
    try:
        vision.expand(pres, patched.token_counts[:2])
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


# ------------------------------------------------------------------ refusal


def arm_refusal() -> None:
    """THE OLD ARM, INVERTED. `vision_seam_unbuilt` was the guarantee that a vision
    presentation could not silently reach Qwen as nothing. It must now be UNREACHABLE for
    an image request — its survival would mean the seam is still not wired."""
    path = ENDPOINT_SOURCE
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    base = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_H3Base"
    )
    condition = next(
        node
        for node in base.body
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
    # THE REFERENCE LATENT ROWS. The PORT dialect builds them; the diffusers dialect wants
    # them interleaved into `hidden_states` in `video_indices` order, which is a different
    # assembly and is unbuilt — and no diffusers-format artifact is bound for it to run
    # against anyway. Named apart so neither claim borrows the other's evidence.
    check(
        "conditioning_rows_unbuilt_diffusers" in src,
        "the diffusers dialect refuses reference rows under its OWN code",
    )
    check(
        "_condition_rows" in src and "video_patch_proj" in src,
        "the port dialect BUILDS one conditioning row block per packed segment",
    )
    check(
        "condition_geometry" in src and "disagree" in src,
        "the derived geometry is checked against the encode rather than trusted",
    )
    check(
        "audio_reference_unpacked" in src and "video_reference_undecoded" in src,
        "audio and video references refuse under their own codes, not the seam's",
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
    from h3_arch.presentation import PresentedReference, build
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
    # A SMALL reference canvas on purpose: the real 2048 rule makes 6,144 tokens per image,
    # which is the right number for a card and the wrong one for a CI arm. The seam is
    # identical; only the grid is smaller.
    from PIL import Image

    small = _image(5, 512, 768).resize((256, 192), Image.Resampling.LANCZOS)
    pres = build(tok, "a fox in the grass", references=(PresentedReference("image", small),))
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
    """OUR vision tower against the OFFICIAL one, destination for destination.

    #540 proved the whole `h3_ref` text encoder key-exact against the official diffusers
    tree. That is the OTHER dialect. This arm asks the question the vision seam actually
    depends on: does the tower inside `h3_arch`'s text encoder — the one the BOUND carrier
    fills — have exactly the destinations `transformers`' own `Qwen3VLVisionModel` has?

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


def arm_refrows() -> None:
    """The reference LATENT rows: what the layout RESERVES and what the patchifier PRODUCES
    must be the same integer, and the anchors' noising must be the reference's.

    This is the arm the old `RefBlock(kind="image")` placeholder would have failed loudly:
    zero latent extents reserved ZERO rows for every reference, so a ref2va layout was
    byte-identical to a t2va one and nothing said so.
    """
    import torch

    from h3_arch import vision
    from h3_arch.dit import patchify_video
    from h3_arch.layout import PackedLayout, RefBlock, latent_grid

    grid = latent_grid(124, 1344, 768)
    patch = (1, 2, 2)

    for width, height in ((2048, 2048), (4096, 2048), (2048, 3072)):
        image = type("I", (), {"size": (width, height)})()
        t, h, w = vision.reference_block_geometry(image)
        check(
            (t, h, w) == (1, height // 16, width // 16),
            f"a {width}x{height} reference encodes to latents {(t, h, w)}",
        )
        layout = PackedLayout(
            32, grid, refs=(RefBlock(kind="image", latent_t=t, latent_h=h, latent_w=w),)
        )
        reserved = sum(end - start for start, end, k in layout.segments if k == "ref_img")
        produced = patchify_video(torch.zeros(1, 24, t, h, w), patch).shape[0]
        check(
            reserved == produced > 0,
            f"the layout reserves {reserved} rows and the patchifier produces {produced}",
        )

    # RED CONTROL: the placeholder that shipped — `RefBlock(kind="image")` with no
    # geometry, one per reference. It does not reserve zero rows and carry on; it divides
    # by a zero sqrt-area and RAISES, inside `prepare`, before any component is leased.
    # So ref2va never reached the text encoder at all, which is what #529's "Ref2VA never
    # ran" recorded from the other end.
    bare = PackedLayout(32, grid)
    try:
        PackedLayout(32, grid, refs=(RefBlock(kind="image"),))
        fail("RED: the shipped zero-geometry RefBlock refuses")
    except ZeroDivisionError:
        observe(
            "RED: the shipped zero-geometry RefBlock raised ZeroDivisionError in the layout",
            "ref2va could never reach the text encoder — it died building the packed sequence",
        )
    block = RefBlock(kind="image", latent_t=1, latent_h=128, latent_w=128)
    real = PackedLayout(32, grid, refs=(block,))
    check(
        real.seq_len > bare.seq_len,
        f"a real reference block DOES grow the sequence: {bare.seq_len} -> {real.seq_len}",
    )

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

    # The CONDITIONING encode is not the target encode: it samples and rounds to fp16.
    import inspect

    from h3_arch.video_vae import AutoencoderKLMiniMaxH3

    src = inspect.getsource(AutoencoderKLMiniMaxH3.encode_condition)
    check("torch.float16" in src, "the conditioning encode rounds the sample to float16")
    check("manual_seed" in src, "the conditioning posterior is SAMPLED under a fixed seed")
    check(
        "torch.clamp(logvar, -30.0, 20.0)" in src,
        "the logvar is clamped the way upstream's DiagonalGaussianDistribution clamps it",
    )
    check(
        "chunk" in inspect.getsource(AutoencoderKLMiniMaxH3.encode)
        and "float16" not in inspect.getsource(AutoencoderKLMiniMaxH3.encode),
        "RED: the TARGET encode still takes the mean and does NOT round — the two differ",
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
    "preprocess": arm_preprocess,
    "presentation": arm_presentation,
    "splices": arm_splices,
    "deepstack": arm_deepstack,
    "towerkeys": arm_towerkeys,
    "forward": arm_forward,
    "refrows": arm_refrows,
    "refusal": arm_refusal,
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
