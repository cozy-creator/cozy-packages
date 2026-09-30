"""Public Anima contracts with actual CPU modules/hooks and Runtime snapshots."""

from functools import wraps
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers import ClassifierFreeGuidance
from diffusers.hooks.first_block_cache import _FBC_LEADER_BLOCK_HOOK
from diffusers.hooks.hooks import ModelHook
from cozy_runtime.author import is_declared_pure, pure
from cozy_runtime.internal import memory, tiers
import anima
from test_stage_scopes import fixture_model


@pytest.mark.parametrize("rope_type", ["default", "dynamic"])
def test_real_pipeline_builder_declares_only_supported_immutable_components(rope_type: str) -> None:
    source = fixture_model().pipe
    document = {
        name: module.config.to_dict() if hasattr(module.config, "to_dict") else dict(module.config)
        for name, module in source.components.items()
    }
    if rope_type == "dynamic":
        document["text_encoder"]["rope_parameters"] = {"rope_type": "dynamic", "factor": 2.0}
    document["scheduler"] = source.scheduler_config
    built = anima.AnimaPipeline(SimpleNamespace(mapping=lambda: document))
    assert is_declared_pure(built.components["transformer"])
    assert is_declared_pure(built.components["text_conditioner"])
    encoder = built.components["text_encoder"]
    assert encoder.rotary_emb.rope_type == rope_type
    assert is_declared_pure(encoder) == (rope_type == "default")
    assert not is_declared_pure(built.components["vae"])
    assert callable(built.components["vae"]._cozy_retry_state[0])
    assert all(not m.training for m in built.components.values())
    assert built.components["vae"].use_tiling


def test_vae_cache_identity_and_runtime_rng_restore_with_real_decode() -> None:
    vae = fixture_model().pipe.components["vae"]
    anima._declare_vae_retry_state(vae)
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "vae", vae, None, tiers.HOST)
    assert owner._state_contract(held, [])
    latent = torch.randn(1, 4, 1, 4, 4)
    with torch.no_grad():
        expected = vae.decode(latent).sample.clone()
    original = torch.ones(2)
    cache, indices = [original, None], [0]
    vae._feat_map, vae._conv_idx = cache, indices
    generator = torch.Generator().manual_seed(71)
    before = torch.get_rng_state().clone()
    before_generator = generator.get_state().clone()
    previous_encoder_cache = vae._enc_feat_map
    snapshot = owner._arm(held, [generator])
    cache[0], indices[0] = torch.zeros(2), 2
    vae._feat_map, vae._enc_feat_map = [], [torch.zeros(2)]
    torch.rand(3)
    torch.rand(3, generator=generator)
    owner._restore(snapshot)
    assert vae._feat_map is cache and cache[0] is original
    assert vae._conv_idx is indices and indices == [0]
    assert vae._enc_feat_map is previous_encoder_cache
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(before_generator, generator.get_state())
    with torch.no_grad():
        actual = vae.decode(latent).sample
    assert torch.equal(expected, actual)


def test_fbc_contract_restores_owned_contexts_and_declines_foreign_hooks() -> None:
    root = fixture_model().pipe.components["transformer"]
    pure(root)
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "trunk", root, None, tiers.HOST)
    original = root.forward
    detach = anima._apply_first_block_cache(root, ClassifierFreeGuidance(), 0.075)
    try:
        assert owner._state_contract(held, [])
        leader = root.transformer_blocks[0]._diffusers_hook.hooks[_FBC_LEADER_BLOCK_HOOK]
        manager = leader.state_manager
        manager.set_context("cond")
        cond = manager.get_state()
        manager.set_context("uncond")
        uncond = manager.get_state()
        old = torch.ones(2)
        cond.head_block_output = old
        cond.head_block_residual = old
        cond.should_compute = True
        uncond.tail_block_residuals = (old,)
        cache = manager._state_cache
        save, restore = root._cozy_retry_state
        snapshot = save()
        indices = leader._metadata._cached_parameter_indices
        leader._metadata._get_parameter_from_args_kwargs("hidden_states", (old,))
        cond.head_block_output = torch.zeros(2)
        cond.head_block_residual = torch.zeros(2)
        cond.should_compute = False
        uncond.tail_block_residuals = None
        manager.set_context("failed")
        manager.get_state()
        restore(snapshot)
        assert manager._state_cache is cache and set(cache) == {"cond", "uncond"}
        assert manager._current_context == "uncond"
        assert cache["cond"] is cond and cache["uncond"] is uncond
        assert cond.head_block_output is old and cond.head_block_residual is old
        assert uncond.tail_block_residuals[0] is old and cond.should_compute
        assert leader._metadata._cached_parameter_indices is indices
        foreign = root.transformer_blocks[0].register_forward_hook(lambda *_: None)
        assert not owner._state_contract(held, [])
        foreign.remove()
        registry = root.transformer_blocks[0]._diffusers_hook
        registry.register_hook(ModelHook(), "foreign")
        assert not owner._state_contract(held, [])
        registry.remove_hook("foreign", recurse=False)
        assert owner._state_contract(held, [])
    finally:
        detach()
    assert root.forward == original and "_cozy_retry_state" not in vars(root)
    assert all(not block._diffusers_hook.hooks for block in root.transformer_blocks)
    assert owner._state_contract(held, [])


def test_unclaimed_forward_is_not_laundered_by_fbc_contract() -> None:
    root = fixture_model().pipe.components["transformer"]
    pure(root)
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "trunk", root, None, tiers.HOST)
    original = root.forward

    @wraps(original)
    def foreign(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    root.forward = foreign
    detach = anima._apply_first_block_cache(root, ClassifierFreeGuidance(), 0.075)
    try:
        assert "_cozy_retry_state" not in vars(root)
        assert not owner._state_contract(held, [])
    finally:
        detach()
    assert root.forward is foreign
