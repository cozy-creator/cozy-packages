"""CPU orchestration proof; GPU correctness is qualified by the Cozy CLI comparison run.

Import the real package and substitute only child calls, checking identity propagation,
preparation overlap, and reuse of one exact original by both upscale families.
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import h3
from cozy_runtime.author import Assets, FileAsset, ImageAsset, Mixed, VideoAsset, describe
from story import StoryReference


class Context:
    request_id = "h3-workflow-proof"
    released = False

    def raise_if_cancelled(self) -> None:
        pass

    def release_gpus(self) -> None:
        self.released = True


class Outputs:
    def __init__(self) -> None:
        self.published: list[tuple[str, Any, str]] = []

    def publish(self, name: str, asset: Any, *, label: str) -> None:
        self.published.append((name, asset, label))


class Telemetry:
    def scope(self, *args: Any, **kwargs: Any) -> Any:
        return nullcontext()

    def log(self, *args: Any, **kwargs: Any) -> None:
        pass


def asset(kind: Any, letter: str) -> Any:
    return kind("sha256:" + letter * 64)


async def references() -> None:
    supplied = asset(ImageAsset, "a")
    authored = asset(ImageAsset, "b")
    cleaned = asset(ImageAsset, "c")
    background_started, caption_started = asyncio.Event(), asyncio.Event()
    calls: list[tuple[str, Any]] = []

    async def remove(**kwargs: Any) -> Any:
        calls.append(("remove", kwargs["image"]))
        background_started.set()
        await caption_started.wait()
        return SimpleNamespace(image=cleaned)

    async def caption(**kwargs: Any) -> Any:
        calls.append(("caption", kwargs["images"]))
        caption_started.set()
        await background_started.wait()
        return SimpleNamespace(captions=[SimpleNamespace(index=0, text="A blue armored pilot with a silver visor.")])

    request = h3.ReferenceRenderInput(
        prompt="<Pilot> and <Robot> guard <Hangar>.", duration_s=15,
        references=[
            StoryReference("Pilot", "character", image=supplied, caption=True, remove_background=True),
            StoryReference("Robot", "character", "A red robot.", image=authored, remove_background=False),
            StoryReference("Hangar", "scene", "An empty hangar.", image=authored),
        ],
    )
    with patch.object(h3, "remove_background", remove), patch.object(h3, "caption_images", caption):
        prepared, media, displayed = await asyncio.wait_for(
            h3._direct_reference_assets(Context(), request, None, Outputs(), Telemetry()), timeout=5,
        )
        assert calls.count(("remove", supplied)) == 1 and len(calls) == 2
        assert ("caption", [supplied]) in calls
        assert "A blue armored pilot with a silver visor." in prepared.prompt
        assert "A red robot." in prepared.prompt
        assert "<Pilot> is the character shown in <Picture 2>" in prepared.prompt
        assert media is not None and len(media) == 3
        assert displayed == [cleaned]
        # A prepared long-form handoff must never run either child again.
        before = list(calls)
        handoff = h3.ReferenceRenderInput(prompt=prepared.prompt, duration_s=15)
        unchanged, same_media, displayed = await h3._direct_reference_assets(Context(), handoff, media, Outputs(), Telemetry())
        assert unchanged is handoff and same_media is media and calls == before
        assert displayed == []
    print("reference preparation overlaps, propagates identity, and runs once: PASS")


async def comparison() -> None:
    original, h3_video, restored = (asset(VideoAsset, c) for c in "def")
    latents, enlarged = (asset(FileAsset, c) for c in "ab")
    calls: list[tuple[str, Any]] = []
    context, output = Context(), Outputs()
    seed = 581203

    async def generate(**kwargs: Any) -> Any:
        calls.append(("generate", kwargs))
        assert kwargs["payload"].retain_latents and kwargs["payload"].seed == seed
        return h3.RetainedVideoOutput(original, latents, [])

    async def upscale(**kwargs: Any) -> Any:
        calls.append(("latent", kwargs))
        assert kwargs == {"latents": latents, "width": 2016, "height": 1152}
        return SimpleNamespace(latents=enlarged)

    async def refine(**kwargs: Any) -> Any:
        calls.append(("refine", kwargs))
        assert kwargs["payload"].latents is enlarged
        assert kwargs["payload"].frames == 362
        return h3.H3VideoOutput(h3_video, [])

    async def restore(**kwargs: Any) -> Any:
        calls.append(("restore", kwargs))
        assert kwargs["video"] is original and kwargs["seed"] == seed
        assert kwargs["resolution"].value == "1080p"
        return SimpleNamespace(video=restored)

    with patch.object(h3, "generate_fl2va_turbo", generate), patch.object(h3, "upscale_latents", upscale), patch.object(h3, "refine_fl2va", refine), patch.object(h3, "restore_video", restore):
        result = await h3._compare_upscalers(
            context, h3.UpscaleComparisonInput(prompt="Mecha fights a monster.", duration_s=15,
                seed=seed, resolutions=(h3.Resolution.FULL_HD,)),
            Assets[Mixed]([]), output, Telemetry(), trunk="fl2va",
        )
    assert [name for name, _ in calls] == ["generate", "latent", "refine", "restore"]
    assert result.original is original and result.h3_latent == [h3_video] and result.seedvr2 == [restored]
    assert result.errors == [] and context.released
    assert [name for name, _, _ in output.published] == ["original", "h3_latent", "seedvr2"]
    print("one retained turbo original feeds both complete upscale chains: PASS")


if __name__ == "__main__":
    surfaces = describe(h3.app)
    assert any(surface.name == "generate_fl2va_turbo" for surface in surfaces)
    print(f"real H3 imported descriptor: {len(surfaces)} entrypoints PASS")
    asyncio.run(references())
    asyncio.run(comparison())
