"""Bounded native image references for independent camera-cut shots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import AssetBound, Image, ImageAsset, InvalidRequest

MAX_IMAGES = 9


class StoryReference(msgspec.Struct, forbid_unknown_fields=True):
    image: Annotated[
        ImageAsset,
        AssetBound(max_bytes=64 << 20, max_decoded_bytes=3 * 16_777_216),
    ]
    description: Annotated[str, msgspec.Meta(min_length=1, max_length=512)]
    subject: Annotated[str, msgspec.Meta(max_length=64)] = ""
    fidelity: Literal["auto", "low", "medium", "high"] = "auto"


class ReferenceFrame(msgspec.Struct):
    image: Annotated[
        ImageAsset,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=3 * 16_777_216,
            media_types=("image/png",),
        ),
    ]
    frame_index: int
    signature: str


class SelectedReference(msgspec.Struct):
    image: ImageAsset
    description: str
    signature: str
    fidelity: Literal["auto", "low", "medium", "high"] = "auto"
    subject: str = ""


def signature(image: Image) -> str:
    """Small RGB thumbnail for deterministic near-duplicate suppression, not quality scoring."""
    return image.convert("RGB").resize((8, 8)).tobytes().hex()


def sample_positions(frames: int) -> tuple[int, ...]:
    """Prefer a clip's middle, then its quarter and three-quarter views; exclude endpoints."""
    return tuple(dict.fromkeys((frames // 2, frames // 4, 3 * frames // 4)))


def _similar(left: str, right: str) -> bool:
    a, b = bytes.fromhex(left), bytes.fromhex(right)
    if len(a) != 192 or len(b) != 192:
        raise InvalidRequest("reference thumbnail signature is invalid", code="reference_signature")
    return sum(abs(x - y) for x, y in zip(a, b, strict=True)) <= 4 * len(a)


def select_references(
    stable: Sequence[SelectedReference],
    history: Sequence[Sequence[ReferenceFrame]],
    *,
    history_frames: int,
) -> list[SelectedReference]:
    """Stable anchors first; recent clips' middle views before additional views.

    The conservative default uses one historical view. Larger budgets are an explicit
    experiment; this policy does not claim blur detection or semantic relevance scoring.
    """
    if not 0 <= history_frames <= 6 or len(stable) > MAX_IMAGES:
        raise InvalidRequest("reference budget exceeds its bound", code="reference_policy")
    selected: list[SelectedReference] = []
    for item in stable:
        existing = next((ref for ref in selected if ref.image.digest == item.image.digest), None)
        if existing is None:
            selected.append(item)
        else:
            # One image can describe multiple subjects; preserve both descriptions.
            selected[selected.index(existing)] = msgspec.structs.replace(
                existing, description=f"{existing.description}; {item.description}"
            )
    remaining = min(history_frames, MAX_IMAGES - len(selected))
    for view in range(3):
        for index in range(len(history) - 1, -1, -1):
            if remaining == 0:
                return selected
            if view >= len(history[index]):
                continue
            candidate = history[index][view]
            if any(
                candidate.image.digest == ref.image.digest
                or _similar(candidate.signature, ref.signature)
                for ref in selected
            ):
                continue
            selected.append(
                SelectedReference(
                    candidate.image,
                    f"Relevant character, prop and setting appearance in shot {index + 1}; "
                    "the current shot description determines the action and camera composition",
                    candidate.signature,
                    subject="The existing cast and world from preceding shots",
                )
            )
            remaining -= 1
    return selected


def shot_prompt(
    shared: str, description: str, references: Sequence[SelectedReference], *, index: int
) -> str:
    groups: dict[str, list[tuple[int, SelectedReference]]] = {}
    for slot, ref in enumerate(references, start=1):
        groups.setdefault(ref.subject or f"Reference {slot}", []).append((slot, ref))
    subjects = []
    for number, (name, items) in enumerate(groups.items(), start=1):
        descriptions = "; ".join(dict.fromkeys(ref.description for _, ref in items))
        pictures = ", ".join(f"<Picture {slot}>" for slot, _ in items)
        subjects.append(f"<Subject {number}>: {name}. {descriptions}. Appearance in {pictures}.")
    parts = [
        "Subject definitions:\n" + "\n".join(subjects) if subjects else "",
        f"Video summary:\n{shared.strip()}" if shared.strip() else "",
        "Retention:\nPreserve the referenced subjects' identity and relevant scene details.",
        f"Shot {index + 1} description:\n{description.strip()}",
    ]
    prompt = "\n\n".join(part for part in parts if part)
    if len(prompt) > 4096:
        raise InvalidRequest(
            f"shot {index + 1}'s shared prompt, reference descriptions and shot exceed 4096 "
            "characters; shorten them before rendering",
            fields=["shots", "prompt", "references"],
        )
    return prompt
