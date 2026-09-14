"""Real CPU Diffusers dispatch checks for diagnostic capture and hook restoration."""

from __future__ import annotations

import torch
from cozy_runtime.author import AttentionLayout, attention_scope
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Attention
from h3_attention_oracle import Capture, Captured, Input


def main() -> None:
    root = torch.nn.Module()
    root.token_refiner = MiniMaxH3Attention(hidden_size=8, heads=1, dim_head=8)
    root.transformer_blocks = torch.nn.ModuleList(
        [MiniMaxH3Attention(hidden_size=8, heads=1, dim_head=8)]
    )
    # This refiner is longer than the old heuristic's 1,024-token cutoff.
    text = torch.randn(1, 1300, 8)
    tokens = torch.randn(1, 4, 8)
    reference = _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE]
    processors = [root.token_refiner.processor, root.transformer_blocks[0].processor]
    defaults = [processor._attention_backend for processor in processors]
    capture = Capture(Input(prompt="CPU capture contract", reference_backend="sdpa"))

    def invoke(_step: object) -> None:
        root.token_refiner(text)
        root.transformer_blocks[0](tokens)

    layout = AttentionLayout(4, 1, 0, dense_until_step=10)
    with torch.no_grad(), attention_scope(layout):
        try:
            capture.sample(root, invoke, lambda _step: None)
        except Captured:
            pass
        else:
            raise AssertionError("the requested DiT block was not captured")
    assert capture.tensors is not None and capture.tensors[0].shape == (1, 4, 1, 8)
    assert capture.layout is layout
    assert capture.module_path == "transformer_blocks.0"
    assert _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE] is reference
    assert [processor._attention_backend for processor in processors] == defaults
    for module in (root.token_refiner, root.transformer_blocks[0]):
        assert not module._forward_hooks and not module._forward_pre_hooks
        assert not hasattr(module, "_cozy_sol_site")
    # An ordinary next call executes normally after the diagnostic early exit.
    assert torch.isfinite(root.transformer_blocks[0](tokens)).all()
    print("real H3 CPU capture, exact module identity and restoration: passed")


if __name__ == "__main__":
    main()
