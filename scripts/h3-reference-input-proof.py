"""Model-free proof of reference selection and prompt composition, not asset custody or GPU quality."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from cozy_runtime.author import ImageAsset, InvalidRequest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
from story import (  # noqa: E402
    StoryReference,
    image_prompt,
    resolve_reference_images,
    select_references,
    shot_prompt,
    validate_references,
)


async def main() -> None:
    supplied = ImageAsset("sha256:" + "a" * 64)
    generated = ImageAsset("sha256:" + "b" * 64)
    hero = StoryReference("Hero", "character", "An adult woman in a teal coat.", image=supplied)
    depot = StoryReference("Depot", "scene", "A wet train platform.")
    calls: list[str] = []

    async def generate(reference: StoryReference) -> ImageAsset:
        calls.append(reference.name)
        return generated

    assert await resolve_reference_images([hero], generate) == {"Hero": supplied}
    assert not calls
    mixed = await resolve_reference_images([hero, depot], generate)
    assert mixed["Hero"] is supplied and mixed["Depot"] is generated
    assert calls == ["Depot"]
    selected = select_references(["HERO", "depot"], validate_references([hero, depot]), index=0)
    text = shot_prompt(
        "Photorealistic.", "{hErO} walks through {DEPOT}.", selected,
        index=0, continuous=True, soundscape="Rain and footsteps.", music="N/A",
    )
    assert hero.description in text and "<Subject 1> walks through <Subject 2>" in text
    sections = (
        "subject_definitions", "summary", "retention_analysis", "detailed_description",
        "overall_soundscape", "non_diegetic_music",
    )
    positions = [text.index(section + ":") for section in sections]
    assert positions == sorted(positions)
    assert text.index("Photorealistic.") < text.index("[Shot 1]\n")
    assert "Continue the preceding" not in text
    assert "Continue the preceding" in shot_prompt(
        "", "{hero} continues walking.", selected, index=1, continuous=True
    )
    assert depot.description in image_prompt(depot)
    for invalid in ([hero, StoryReference("HERO", "scene", "Duplicate")], [StoryReference("empty", "scene")]):
        try:
            validate_references(invalid)
        except InvalidRequest:
            pass
        else:
            raise AssertionError("invalid reference accepted")
    try:
        select_references(["Hero", "HERO"], validate_references([hero]), index=0)
    except InvalidRequest:
        pass
    else:
        raise AssertionError("duplicate selection accepted")

    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def failing(reference: StoryReference) -> ImageAsset:
        if reference.name == "Depot":
            await entered.wait()
            raise ValueError("generation failed")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    try:
        await resolve_reference_images([depot, StoryReference("Other", "scene", "Garden")], failing)
    except ValueError:
        pass
    else:
        raise AssertionError("generation failure swallowed")
    assert cancelled.is_set()
    print("mixed references, case-insensitive names, six-section prompts and sibling cancellation: PASS")


if __name__ == "__main__":
    asyncio.run(main())
