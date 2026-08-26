"""The H3 PRESENTATION — how a request becomes the token sequence Qwen3-VL sees.

`prompt` on the wire is text. Optional first/last keyframes also enter Qwen3-VL as image
blocks and separately enter the video VAE as target-clock anchors. This file owns the
presentation half, which is pure CPU string and integer work with no weights in it.

The presentation is NOT chat-templated — no `<|im_start|>`, no system turn, no thinking
block. Token ids are the raw prompt and label text with vision blocks spliced in, and the
labels are ordinal per TYPE in request order:

    text-only   <prompt>
    keyframed   "<Picture 1>: " <vision> ["<Picture 2>: " <vision>] <prompt>

TOKEN TAGS. The DiT modulates by modality tag, and a vision block inside the text span
carries the VIDEO tag (0) rather than the text tag (1), flanking `<|vision_start|>` and
`<|vision_end|>` included. That is why the text segment is the one packed segment that
splits into runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

#: Qwen3-VL's vision sentinels and its pad token. Ids, not names: the vocabulary is
#: bundled with this endpoint and these are facts about it.
VISION_START = 151652
VISION_END = 151653
PAD_TOKEN = 151643

#: Modality tags the DiT's AdaLN indexes by.
TAG_VIDEO = 0
TAG_TEXT = 1

#: The bundled vocabulary — this endpoint's own asset, exactly like the model library it
#: imports. Not an artifact identifier and not a catalog ref.
_TOKENIZER = Path(__file__).resolve().parent.parent / "tokenizer"


@dataclass(frozen=True, slots=True)
class VisionBlock:
    """One keyframe image block spliced into the Qwen token sequence."""

    pixels: Any


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


def build(
    tokenizer: Encoder,
    prompt: str,
    *,
    keyframes: tuple[Any, ...] = (),
) -> Presentation:
    """Build the retained text plus optional first/last-keyframe presentation."""
    rows: list[Row] = []
    tags: list[int] = []

    def text(value: str) -> None:
        for tid in tokenizer.ids(value):
            rows.append(tid)
            tags.append(TAG_TEXT)

    def vision(pixels: Any) -> None:
        # the sentinels carry the VIDEO tag with the block they flank
        rows.append(VISION_START)
        tags.append(TAG_VIDEO)
        rows.append(VisionBlock(pixels))
        tags.append(TAG_VIDEO)
        rows.append(VISION_END)
        tags.append(TAG_VIDEO)

    for index, frame in enumerate(keyframes):
        text(f"<Picture {index + 1}>: ")
        vision(frame)

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
