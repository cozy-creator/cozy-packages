"""CPU conformance for real CLIP output capture and a mid-forward allocation refusal."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import CLIPTextConfig, CLIPTextModel

from cozy_runtime.author import is_declared_pure, pure
from cozy_runtime.internal import memory, tiers
from sdxl.capture import declare_output_capture


class RefuseAfterCapturedLayer(TorchDispatchMode):  # type: ignore[misc]  # Torch is optional in static checks.
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.captured_before_refusal = 0

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ) -> Any:
        self.calls += 1
        capture = importlib.import_module("transformers.utils.output_capturing")
        collected = capture._active_collector.get()
        captured = len(collected.get("hidden_states", ())) if collected is not None else 0
        if captured >= 2:
            self.captured_before_refusal = captured
            raise torch.OutOfMemoryError("injected CLIP allocation refusal")
        return func(*args, **(kwargs or {}))


def main() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = CLIPTextConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=16,
        attention_dropout=0.0, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    model = CLIPTextModel(config).eval()
    pure(model)
    declare_output_capture(model)
    planner = memory.Planner(torch, torch.device("cpu"))
    held = planner.adopt("clip", "text_encoder", model, None, tiers.HOST)
    assert planner._state_contract(held, [])
    hooks = tuple(hook for module in model.modules() for hook in module._forward_hooks.values())
    assert hooks and all(is_declared_pure(hook) for hook in hooks)
    declare_output_capture(model)
    assert hooks == tuple(hook for module in model.modules() for hook in module._forward_hooks.values())
    inputs = torch.tensor([[1, 4, 7, 2]], dtype=torch.long)
    capture = importlib.import_module("transformers.utils.output_capturing")
    with torch.inference_mode():
        expected = model(inputs, output_hidden_states=True)
        fault = RefuseAfterCapturedLayer()
        try:
            with fault:
                model(inputs, output_hidden_states=True)
        except torch.OutOfMemoryError:
            pass
        else:
            raise AssertionError("fault was not reached")
        assert fault.captured_before_refusal >= 2
        assert capture._active_collector.get() is None
        retried = model(inputs, output_hidden_states=True)
    assert len(expected.hidden_states) == len(retried.hidden_states) == 3
    assert all(torch.equal(a, b) for a, b in zip(expected.hidden_states, retried.hidden_states, strict=True))
    assert torch.equal(expected.last_hidden_state, retried.last_hidden_state)
    assert capture._active_collector.get() is None and planner._state_contract(held, [])

    def foreign(module: Any, args: Any, output: Any) -> None:
        return None

    handle = model.register_forward_hook(foreign)
    try:
        declare_output_capture(model)
        assert not is_declared_pure(foreign) and not planner._state_contract(held, [])
    finally:
        handle.remove()
    # The optional capture module can be absent in an older Transformers cohort.
    with patch.dict("sys.modules", {"transformers.utils.output_capturing": None}):
        declare_output_capture(model)
    print("CLIP capture: exact callbacks declared; unknown hook refused; OOM collector reset; retry bit-identical")


if __name__ == "__main__":
    main()
