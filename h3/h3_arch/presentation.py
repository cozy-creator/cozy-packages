"""The H3 PRESENTATION — how a request becomes the token sequence Qwen3-VL sees.

`prompt` on the wire is text. The conditioning operation is not: H3 builds a multimodal
presentation and separately VAE-encodes visual and audio conditions, and the two products
meet later in the packed sequence. This file owns the first half, which is pure CPU string
and integer work with no weights in it at all.

The presentation is NOT chat-templated — no `<|im_start|>`, no system turn, no thinking
block. Token ids are the raw prompt and label text with vision blocks spliced in, and the
labels are ordinal per TYPE in request order:

    t2va    <prompt>
    fl2va   "<Picture 1>: " <vision> ["<Picture 2>: " <vision>] <prompt>
    ref2va  image  ->  "<Picture i>: " <vision>
            audio  ->  "<Audio j>: "                (a waveform NEVER enters Qwen)
            video  ->  "<Video k>: " then per 2-frame block "<T.T seconds>" <vision>
            then <prompt>

An audio reference contributes a LABEL and nothing else here; its waveform goes to the
audio VAE. That asymmetry is the reason `<Audio j>` exists at all — without it the
presentation would not record that the reference was there.

TOKEN TAGS. The DiT modulates by modality tag, and a vision block inside the text span
carries the VIDEO tag (0) rather than the text tag (1), flanking `<|vision_start|>` and
`<|vision_end|>` included. That is why the text segment is the one packed segment that
splits into runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

#: Qwen3-VL's vision sentinels and its pad token. Ids, not names: the vocabulary is
#: bundled with this endpoint and these are facts about it.
VISION_START = 151652
VISION_END = 151653
PAD_TOKEN = 151643

#: Modality tags the DiT's AdaLN indexes by.
TAG_VIDEO = 0
TAG_TEXT = 1

#: Qwen3-VL's own image normalization. Not a preference: the vision tower was trained
#: against it, and a different mean/std is a different model.
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)

#: A video reference is presented to Qwen at roughly 2 fps in 2-frame temporal patches,
#: while the video VAE separately encodes the SAME reference on H3's own 24 fps clock.
#: Two rates for one asset, on purpose.
PRESENTATION_FPS = 2.0

#: The bundled vocabulary — this endpoint's own asset, exactly like the model library it
#: imports. Not an artifact identifier and not a catalog ref.
_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"


@dataclass(frozen=True, slots=True)
class VisionBlock:
    """One spliced vision block: the pixels Qwen sees and whether they are a 2-frame video
    patch (temporal patch filled by two real frames) or a still (frame repeated)."""

    pixels: Any
    video_block: bool


#: A presentation row is either a token id or a vision block standing in for one.
Row = int | VisionBlock


@dataclass(frozen=True, slots=True)
class Presentation:
    rows: tuple[Row, ...]
    tags: tuple[int, ...]
    """One tag per PRESENTATION row. Expands with the vision block at conditioning time."""

    def text_ids(self) -> tuple[int, ...]:
        return tuple(r for r in self.rows if isinstance(r, int))


#: The smallest vocabulary the bundled files can honestly produce. The released
#: `vocab.json` carries 151,643 pieces and the special-token block takes it past that; a
#: tokenizer that constructs with fewer has not read them. This is a floor on a COUNT, not
#: a version check — it is true of every Qwen3-VL vocabulary this endpoint could be given.
_MIN_VOCAB = 151_000


class Tokenizer:
    """The bundled Qwen2-family BPE vocabulary, built from its own two files.

    Deliberately NOT `from_pretrained`: given a string that is not a directory that
    spelling resolves against the Hub, so it is a fetch this endpoint might one day make by
    accident, and `fence.py::no-identifiers-in-code` refuses it for exactly that reason.

    IT PROVES IT LOADED, at construction, because the failure mode is SILENT and shipped.
    `Qwen2Tokenizer` took `vocab_file=` / `merges_file=` before transformers 5 and takes
    `vocab=` / `merges=` after it; the old spelling does not raise on the new library, it
    lands in `**kwargs` and is discarded, and what constructs is a tokenizer holding TWO
    pieces that encodes every prompt to the empty list. Downstream, `build`'s empty-rows
    fallback turned that into one pad token, so the endpoint conditioned every generation
    on NO PROMPT AT ALL and reported nothing. A vocabulary that did not load is a refusal
    here rather than a garbage render later.
    """

    def __init__(self) -> None:
        from transformers import Qwen2Tokenizer

        settings = json.loads((_TOKENIZER / "tokenizer_config.json").read_text())
        self._tok = Qwen2Tokenizer(
            vocab=str(_TOKENIZER / "vocab.json"),
            merges=str(_TOKENIZER / "merges.txt"),
            unk_token=settings["unk_token"],
            bos_token=None,
            eos_token=settings["eos_token"],
            pad_token=settings["pad_token"],
        )
        size = len(self._tok)
        if size < _MIN_VOCAB:
            raise RuntimeError(
                f"the bundled vocabulary did not load: the tokenizer holds {size} pieces "
                f"and the released vocabulary has at least {_MIN_VOCAB}. Every prompt would "
                "encode to the empty list and this endpoint would condition on nothing"
            )

    def ids(self, text: str) -> list[int]:
        out: list[int] = self._tok(text, add_special_tokens=False)["input_ids"]
        return out


class Encoder(Protocol):
    """What `build` actually needs: one method turning text into ids. Written as a protocol
    rather than as `Tokenizer` because the ONE thing that can go wrong here is a tokenizer
    that encodes nothing, and a red control proving the refusal fires has to be spellable
    without constructing 151,645 pieces of vocabulary to do it."""

    def ids(self, text: str) -> list[int]: ...


ReferenceKind = Literal["image", "audio", "video"]


@dataclass(frozen=True, slots=True)
class PresentedReference:
    """One ordered reference as the PRESENTATION sees it. A soundtracked video appears
    here once, as a video: its waveform is the audio VAE's business."""

    kind: ReferenceKind
    pixels: Any = None
    """[T, H, W, C] for a video sampled at `PRESENTATION_FPS`, [1, H, W, C] for an image."""
    timestamps: tuple[float, ...] = ()


def build(
    tokenizer: Encoder,
    prompt: str,
    *,
    keyframes: tuple[Any, ...] = (),
    references: tuple[PresentedReference, ...] = (),
) -> Presentation:
    """The one presentation builder for every retained workflow.

    v1 shipped two disjoint layouts behind a hard either/or dispatcher, so a keyframe was
    never consumed alongside a reference. This is one layout, and combined keyframe +
    reference requests are a DEPLOYMENT-GATED door (`RefServeSettings.combined_keyframes`)
    because their layout is proven and their OUTPUT never was.
    """
    rows: list[Row] = []
    tags: list[int] = []

    def text(value: str) -> None:
        for tid in tokenizer.ids(value):
            rows.append(tid)
            tags.append(TAG_TEXT)

    def vision(pixels: Any, *, video_block: bool) -> None:
        # the sentinels carry the VIDEO tag with the block they flank
        rows.append(VISION_START)
        tags.append(TAG_VIDEO)
        rows.append(VisionBlock(pixels, video_block))
        tags.append(TAG_VIDEO)
        rows.append(VISION_END)
        tags.append(TAG_VIDEO)

    if references:
        counters = {"image": 0, "audio": 0, "video": 0}
        for ref in references:
            counters[ref.kind] += 1
            if ref.kind == "image":
                text(f"<Picture {counters['image']}>: ")
                vision(ref.pixels, video_block=False)
            elif ref.kind == "audio":
                text(f"<Audio {counters['audio']}>: ")
            else:
                frames, stamps = _even_frames(ref)
                text(f"<Video {counters['video']}>: ")
                for i in range(0, len(stamps), 2):
                    text(f"<{(stamps[i] + stamps[i + 1]) / 2.0:.1f} seconds>")
                    vision(frames[i : i + 2], video_block=True)
    for index, frame in enumerate(keyframes):
        text(f"<Picture {index + 1}>: ")
        vision(frame, video_block=False)

    before = len(rows)
    text(prompt)
    if prompt and len(rows) == before:
        # THE FALLBACK BELOW IS FOR AN EMPTY REQUEST, NEVER FOR A BROKEN TOKENIZER. It used
        # to catch both, and catching both is how a vocabulary that never loaded became a
        # render conditioned on one pad token with no error anywhere. A prompt with
        # characters in it that produces no rows is a defect upstream of this function.
        raise RuntimeError(
            f"a {len(prompt)}-character prompt produced no tokens: the presentation would "
            "carry no conditioning at all, which is a broken tokenizer and not an empty "
            "request"
        )
    if not rows:
        rows.append(PAD_TOKEN)
        tags.append(TAG_TEXT)
    return Presentation(tuple(rows), tuple(tags))


def _even_frames(ref: PresentedReference) -> tuple[Any, list[float]]:
    """The temporal patch is 2 frames wide, so an odd sample count repeat-pads its last."""
    frames = ref.pixels
    count = int(frames.shape[0])
    stamps = list(ref.timestamps) or [i / PRESENTATION_FPS for i in range(count)]
    if count % 2 == 1:
        import torch

        frames = torch.cat([frames, frames[-1:]], dim=0)
        stamps = [*stamps, stamps[-1]]
    return frames, stamps


def expand_tags(presentation: Presentation, block_sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Presentation tags -> one tag per EMBEDDING row, after each vision block expands.

    A block's merged token count is a fact of its resolved grid, which only the vision
    tower can state, so it arrives here rather than being guessed."""
    out: list[int] = []
    block = 0
    for row, tag in zip(presentation.rows, presentation.tags, strict=True):
        if isinstance(row, VisionBlock):
            out.extend([TAG_VIDEO] * block_sizes[block])
            block += 1
        else:
            out.append(tag)
    if block != len(block_sizes):
        raise ValueError(
            f"the presentation has {block} vision blocks and {len(block_sizes)} sizes were "
            "supplied — the tag expansion would silently misalign the text span"
        )
    return tuple(out)
