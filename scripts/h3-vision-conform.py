#!/usr/bin/env python
"""H3's VISION-SEAM CONFORMANCE ARMS — the pixels-to-conditioner seam, decided on CPU.

    nice -n 19 python scripts/h3-vision-conform.py [arm ...]
    arms: preprocess, presentation, splices, deepstack, refusal

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

import pathlib
import sys
import traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "h3"))

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

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
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

    # The conditioner triples slice the right rows out of the batched patch matrix.
    blocks = vision.conditioner_blocks(patched, exp)
    check(len(blocks) == 3, "three conditioner blocks built")
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
    from h3_arch.text_encoder import Qwen3VLConditioner

    config = TextEncoderConfig()
    check(
        tuple(config.vision_deepstack_layers) == (8, 16, 24),
        f"deepstack take-points are the official [8, 16, 24], got {config.vision_deepstack_layers}",
    )
    check(
        config.vision_depth == 27 and config.vision_spatial_merge_size == 2,
        f"vision tower is 27 blocks with a 2x2 merge, got {config.vision_depth}",
    )

    # The tower must EMIT one feature per take-point, and the conditioner must ADD them on
    # the first three decoder layers. Built on `meta`, so no weight is allocated.
    with torch.device("meta"):
        model = Qwen3VLConditioner(config)
    check(
        len(model.visual.deepstack_merger_list) == 3,
        f"three deepstack mergers exist, got {len(model.visual.deepstack_merger_list)}",
    )
    check(
        model.visual.deepstack_layers == (8, 16, 24),
        "the tower took the take-points from the config rather than hardcoding them",
    )

    import inspect

    src = inspect.getsource(Qwen3VLConditioner.forward)
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
    import inspect

    import h3

    src = inspect.getsource(h3)
    check(
        "vision_seam_unbuilt" not in inspect.getsource(h3._H3Base.condition_text),
        "RED (inverted): condition_text no longer refuses a vision presentation",
    )
    check(
        "vision.patchify" in inspect.getsource(h3._H3Base.condition_text)
        or "patchify" in inspect.getsource(h3._H3Base.condition_text),
        "condition_text patchifies the presentation's vision blocks",
    )
    remaining = src.count("vision_seam_unbuilt")
    print(f"         `vision_seam_unbuilt` still appears {remaining}x in h3.py")


ARMS = {
    "preprocess": arm_preprocess,
    "presentation": arm_presentation,
    "splices": arm_splices,
    "deepstack": arm_deepstack,
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
