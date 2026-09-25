"""Named, fixed character and scene references for independent camera-cut shots."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Annotated, Literal

import msgspec
from cozy_runtime.author import InvalidRequest

MAX_IMAGES = 9
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,47}\Z")
_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_-]*)\}")


class StoryReference(msgspec.Struct, forbid_unknown_fields=True):
    """One image to generate before the first video shot; names are request-local IDs."""

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=48)]
    kind: Literal["character", "scene"]
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=1024)]
    seed: Annotated[int, msgspec.Meta(ge=0, le=9223372036854775807)] | None = None


def reference_seed(reference: StoryReference, request_id: str) -> int:
    if reference.seed is not None:
        return reference.seed
    identity = f"{request_id}/h3/reference/{reference.name}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


def image_prompt(reference: StoryReference) -> str:
    if reference.kind == "character":
        return (
            f"{reference.prompt.strip()}\n"
            "A single full-body character reference portrait, clear face and clothing, "
            "neutral standing pose with visible hands and feet, on a plain white studio "
            "background. One subject, one view, no panels or labels."
        )
    return (
        f"{reference.prompt.strip()}\n"
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
        if reference.name in by_name:
            raise InvalidRequest(
                f"duplicate reference name: {reference.name}", fields=["references"]
            )
        if not reference.prompt.strip():
            raise InvalidRequest(
                f"reference {reference.name} has an empty prompt", fields=["references"]
            )
        by_name[reference.name] = reference
    return by_name


def select_references(
    names: Sequence[str], by_name: dict[str, StoryReference], *, index: int
) -> list[StoryReference]:
    if not 1 <= len(names) <= MAX_IMAGES or len(set(names)) != len(names):
        raise InvalidRequest(
            f"shot {index + 1} must select one to nine distinct reference names", fields=["shots"]
        )
    missing = [name for name in names if name not in by_name]
    if missing:
        raise InvalidRequest(
            f"shot {index + 1} selects unknown references: {', '.join(missing)}", fields=["shots"]
        )
    return [by_name[name] for name in names]


def shot_prompt(
    shared: str, description: str, references: Sequence[StoryReference], *, index: int
) -> str:
    labels = {reference.name: f"<Subject {slot}>" for slot, reference in enumerate(references, 1)}

    def substitute(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in labels:
                raise InvalidRequest(
                    f"shot {index + 1} mentions {{{name}}} without selecting that reference",
                    fields=["shots", "prompt"],
                )
            return labels[name]

        return _PLACEHOLDER.sub(replace, text.strip())

    if not description.strip():
        raise InvalidRequest(f"shot {index + 1} has an empty prompt", fields=["shots"])
    subjects, retention = [], []
    for slot, reference in enumerate(references, 1):
        label = labels[reference.name]
        subjects.append(
            f"{label} is {reference.name}, the {reference.kind} in <Picture {slot}>: "
            f"{reference.prompt.strip()}"
        )
        preserved = (
            "identity, face, body proportions, hair and clothing; the portrait's white "
            "background and pose are not part of this subject"
            if reference.kind == "character"
            else "the environment's architecture, materials and defining features; "
            "its photographed viewpoint and framing are not part of this subject"
        )
        retention.append(f"{label} (appears in [Shot 1]): fully_preserved - {preserved}.")
    prompt = "\n\n".join(
        (
            "subject_definitions:\n" + "\n".join(subjects),
            "summary:\n[reference generation] " + substitute(shared),
            "retention_analysis:\n" + "\n".join(retention),
            "detailed_description:\n[Shot 1]\n"
            "One continuous shot. Compose a new camera view and animate the subjects "
            "according to this shot description:\n" + substitute(description),
        )
    )
    if len(prompt) > 4096:
        raise InvalidRequest(
            f"shot {index + 1}'s shared prompt, reference descriptions and shot exceed 4096 "
            "characters; shorten them before rendering",
            fields=["shots", "prompt", "references"],
        )
    return prompt
