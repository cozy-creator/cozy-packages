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
    roles: tuple[tuple[str, str], ...] = ()


def reference_roles(reference: SelectedReference) -> tuple[tuple[str, str], ...]:
    return reference.roles or ((reference.subject, reference.description),)


def signature(image: Image) -> str:
    """Small RGB thumbnail for deterministic near-duplicate suppression, not quality scoring."""
    return signature_rgb(image.width, image.height, image.convert("RGB").tobytes())


def signature_rgb(width: int, height: int, rgb: bytes) -> str:
    result = bytearray()
    for y in range(8):
        for x in range(8):
            offset = (
                min(height - 1, (2 * y + 1) * height // 16) * width
                + min(width - 1, (2 * x + 1) * width // 16)
            ) * 3
            result.extend(rgb[offset : offset + 3])
    return result.hex()


def history_description(index: int) -> str:
    return (
        f"Relevant character, prop and setting appearance in preceding shot {index + 1}; "
        "the current shot description determines the action and camera composition"
    )


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
            if existing.fidelity != item.fidelity:
                raise InvalidRequest(
                    "the same image has conflicting fidelity settings", fields=["references"]
                )
            # One native Picture can define several Subjects without duplicating its
            # conditioning rows or increasing its relative weight in the reference set.
            selected[selected.index(existing)] = msgspec.structs.replace(
                existing,
                roles=tuple(dict.fromkeys((*reference_roles(existing), *reference_roles(item)))),
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
                    history_description(index),
                    candidate.signature,
                    subject="The existing cast and world from preceding shots",
                )
            )
            remaining -= 1
    return selected


def shot_prompt(
    shared: str, description: str, references: Sequence[SelectedReference], *, index: int
) -> str:
    groups: dict[str, list[tuple[int, str]]] = {}
    for slot, ref in enumerate(references, start=1):
        for subject, details in reference_roles(ref):
            groups.setdefault(subject or f"Reference {slot}", []).append((slot, details))
    subjects = []
    for number, (name, items) in enumerate(groups.items(), start=1):
        descriptions = "; ".join(dict.fromkeys(details for _, details in items))
        pictures = ", ".join(dict.fromkeys(f"<Picture {slot}>" for slot, _ in items))
        subjects.append(f"<Subject {number}>: {name}. {descriptions}. Appearance in {pictures}.")
    parts = [
        "subject_definitions:\n" + "\n".join(subjects) if subjects else "",
        f"summary:\n{'[reference generation] ' if references else ''}{shared.strip()}",
        "retention_analysis:\nUse the reference descriptions for relevant identity and "
        "appearance. The current shot determines who is present, camera, pose, action, "
        "wardrobe changes and location changes."
        if references
        else "",
        f"detailed_description:\n[Shot 1]\n{description.strip()}",
    ]
    prompt = "\n\n".join(part for part in parts if part)
    if len(prompt) > 4096:
        raise InvalidRequest(
            f"shot {index + 1}'s shared prompt, reference descriptions and shot exceed 4096 "
            "characters; shorten them before rendering",
            fields=["shots", "prompt", "references"],
        )
    return prompt
