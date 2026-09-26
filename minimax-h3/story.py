"""Named, fixed character and scene references for independent camera-cut shots."""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import AssetBound, ImageAsset, InvalidRequest

MAX_IMAGES = 9
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_-]*)\}")


class StoryReference(msgspec.Struct, forbid_unknown_fields=True):
    """A supplied image or an image generated from its shared visual description."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    kind: Literal["character", "scene"]
    description: Annotated[str, msgspec.Meta(max_length=1024)] = ""
    image: Annotated[
        ImageAsset | None,
        AssetBound(
            max_bytes=64 << 20,
            max_decoded_bytes=256 << 20,
            media_types=("image/png", "image/jpeg", "image/webp"),
        ),
    ] = None
    seed: Annotated[int, msgspec.Meta(ge=0, le=9007199254740991)] | None = None


def reference_seed(reference: StoryReference, request_id: str) -> int:
    if reference.seed is not None:
        return reference.seed
    identity = f"{request_id}/h3/reference/{reference.name.casefold()}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


async def resolve_reference_images(
    references: Sequence[StoryReference],
    generate: Callable[[StoryReference], Awaitable[ImageAsset]],
) -> dict[str, ImageAsset]:
    """Reuse granted handles and generate only missing images, draining failed siblings."""
    images = {ref.name: ref.image for ref in references if ref.image is not None}
    pending = [ref for ref in references if ref.image is None]

    async def one(reference: StoryReference) -> ImageAsset:
        return await generate(reference)

    tasks = [asyncio.create_task(one(reference)) for reference in pending]
    try:
        generated = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    images.update((ref.name, image) for ref, image in zip(pending, generated, strict=True))
    return images


def image_prompt(reference: StoryReference) -> str:
    if reference.kind == "character":
        return (
            f"{reference.description.strip()}\n"
            "A full-body subject reference with clear identifying features, a natural "
            "neutral pose, and the entire subject visible against a plain white studio "
            "background. One subject, one view, no panels or labels."
        )
    return (
        f"{reference.description.strip()}\n"
        "An environment reference image showing the architecture, materials and defining "
        "features of the location, without people, labels or panels."
    )


def validate_references(references: Sequence[StoryReference]) -> dict[str, StoryReference]:
    if not 1 <= len(references) <= MAX_IMAGES:
        raise InvalidRequest("declare between one and nine references", fields=["references"])
    by_name: dict[str, StoryReference] = {}
    for reference in references:
        if not _NAME.fullmatch(reference.name):
            raise InvalidRequest(
                "reference names must start with a letter and use letters, digits, _ or -",
                fields=["references"],
            )
        key = reference.name.casefold()
        if key in by_name:
            raise InvalidRequest(
                f"duplicate reference name: {reference.name}", fields=["references"]
            )
        if reference.image is None and not reference.description.strip():
            raise InvalidRequest(
                f"reference {reference.name} needs an image or a nonempty description",
                fields=["references"],
            )
        by_name[key] = reference
    return by_name


def select_references(
    names: Sequence[str], by_name: dict[str, StoryReference], *, index: int
) -> list[StoryReference]:
    keys = [name.casefold() for name in names]
    if not 1 <= len(names) <= MAX_IMAGES or len(set(keys)) != len(names):
        raise InvalidRequest(
            f"shot {index + 1} must select one to nine distinct reference names", fields=["shots"]
        )
    missing = [name for name in names if name.casefold() not in by_name]
    if missing:
        raise InvalidRequest(
            f"shot {index + 1} selects unknown references: {', '.join(missing)}", fields=["shots"]
        )
    return [by_name[key] for key in keys]


def shot_prompt(
    style: str,
    description: str,
    references: Sequence[StoryReference],
    *,
    index: int,
    continuous: bool = False,
    soundscape: str = "",
    music: str = "",
) -> str:
    labels = {
        reference.name.casefold(): f"<Subject {slot}>"
        for slot, reference in enumerate(references, 1)
    }

    def substitute(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name.casefold() not in labels:
                raise InvalidRequest(
                    f"shot {index + 1} mentions {{{name}}} without selecting that reference",
                    fields=["shots", "prompt"],
                )
            return labels[name.casefold()]

        return _PLACEHOLDER.sub(replace, text.strip())

    if not description.strip():
        raise InvalidRequest(f"shot {index + 1} has an empty prompt", fields=["shots"])
    subjects: list[str] = []
    retention: list[str] = []
    for slot, reference in enumerate(references, 1):
        label = labels[reference.name.casefold()]
        subjects.append(
            f"{label} is {reference.name}, the {reference.kind} shown in <Picture {slot}>."
            + (f" {reference.description.strip()}" if reference.description.strip() else "")
        )
        preserved = (
            "identity, face, body proportions, hair and clothing; the portrait's white "
            "background and pose are not part of this subject"
            if reference.kind == "character"
            else "the environment's architecture, materials and defining features; "
            "its photographed viewpoint and framing are not part of this subject"
        )
        retention.append(f"{label} (appears in [Shot 1]): fully_preserved - {preserved}.")
    if continuous and index > 0:
        direction = (
            "Continue the preceding action and camera motion seamlessly; preserve the same "
            "environment and soundscape, with no cut or establishing view. "
        )
    elif continuous:
        direction = "Begin one continuous camera take. Establish the opening composition and action. "
    else:
        direction = "One continuous shot. Compose a new camera view and animate the subjects. "
    prompt = "\n\n".join(
        (
            "subject_definitions:\n" + "\n".join(subjects),
            "summary:\n[reference generation] Generate the described shot using "
            + ", ".join(labels.values())
            + " as visual references for the defined subjects and environment.",
            "retention_analysis:\n" + "\n".join(retention),
            "detailed_description:\n"
            + (substitute(style) + "\n" if style.strip() else "")
            + "[Shot 1]\n"
            + direction
            + "Follow this shot description:\n"
            + substitute(description),
            "overall_soundscape:\n"
            + substitute(
                soundscape
                or "Follow the ambience and physical sounds specified in the shot "
                "description, synchronized with the visible action."
            ),
            "non_diegetic_music:\n"
            + substitute(
                music
                or "No background score unless explicitly requested in the shot description."
            ),
        )
    )
    if len(prompt) > 4096:
        raise InvalidRequest(
            f"shot {index + 1}'s style, audio, reference descriptions and shot exceed 4096 "
            "characters; shorten them before rendering",
            fields=["shots", "style", "soundscape", "music", "references"],
        )
    return prompt
