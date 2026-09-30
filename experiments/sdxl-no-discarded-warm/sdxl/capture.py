"""Declare the Transformers output-capture callbacks owned by SDXL's CLIP encoders."""

from __future__ import annotations

import importlib
from types import CodeType
from typing import Any

import cozy_runtime.author as author


def declare_output_capture(model: Any) -> None:
    """Install capture before Runtime adoption and declare only the factory's callbacks.

    Transformers collects transient outputs in a context-local object and resets that
    context in ``finally`` around each forward. Its callbacks do not modify weights or
    inputs. Eager installation keeps the first managed call's contract stable. Older
    Transformers without this mechanism keep their existing behavior.
    """
    mark = getattr(author, "pure", None)
    if mark is None:
        return
    try:
        capture = importlib.import_module("transformers.utils.output_capturing")
    except ModuleNotFoundError as exc:
        if exc.name != "transformers.utils.output_capturing":
            raise
        return
    install = getattr(capture, "maybe_install_capturing_hooks", None)
    factory = getattr(capture, "install_output_capuring_hook", None)
    code = getattr(factory, "__code__", None)
    if install is None or code is None:
        return
    # Function-body identity ties this declaration to the imported factory, not a
    # module/name whitelist that could bless an unrelated existing hook.
    callbacks = {
        constant for constant in code.co_consts
        if isinstance(constant, CodeType) and constant.co_name == "output_capturing_hook"
    }
    if not callbacks:
        return
    before = {id(hook) for module in model.modules() for hook in module._forward_hooks.values()}
    install(model)
    for module in model.modules():
        for hook in module._forward_hooks.values():
            if (
                id(hook) not in before
                and any(getattr(hook, "__code__", None) is body for body in callbacks)
                and getattr(hook, "__globals__", None) is getattr(factory, "__globals__", None)
            ):
                mark(hook)
