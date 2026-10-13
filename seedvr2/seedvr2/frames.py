"""Upstream temporal padding (numz `pad_video_temporal`) over any frame sequence."""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def extend(frames: list[T], count: int, *, prepend: bool = False) -> list[T]:
    """Mirror `count` frames onto one end without repeating the edge frame; a clip too short
    to mirror also repeats its last frame."""
    if count <= 0:
        return frames
    if count >= len(frames):
        repeated = [frames[-1]] * (count - len(frames) + 1)
        mirror = frames[1:][::-1]
        return repeated + mirror + frames if prepend else frames + mirror + repeated
    mirror = frames[1 : count + 1][::-1] if prepend else frames[-count - 1 : -1][::-1]
    return mirror + frames if prepend else frames + mirror
