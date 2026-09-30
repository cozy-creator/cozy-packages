"""Actual Diffusers VAE decoding and Runtime conversion bookkeeping, CPU only."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers import AutoencoderKL

import sdxl
from cozy_runtime.author import pure
from cozy_runtime.internal import memory, tiers
from legacy_decode import _decode_vae as legacy_decode


def fixture() -> tuple[Any, Any, torch.Tensor]:
    torch.manual_seed(61)
    vae = AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,), layers_per_block=1, latent_channels=4,
        norm_num_groups=4, sample_size=8, force_upcast=True,
    ).eval().half()
    pure(vae)
    pipe = SimpleNamespace(components={"vae": vae}, vae_scale=0.18215, _vae_execution_dtype=None)
    model = SimpleNamespace(pipe=pipe)
    latent = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(81)).half()
    return model, vae, latent


@pytest.mark.parametrize("tile", [0, 16])
def test_real_repeated_decode_is_exact_to_frozen_reference(tile: int) -> None:
    candidate, vae, latent = fixture()
    reference = deepcopy(candidate)
    state = torch.get_rng_state().clone()
    for each in (latent, latent + 0.125, latent * 0.5):
        expected = legacy_decode(reference, each, tile)
        actual = sdxl.SdxlModel._decode_vae(candidate, each, tile)
        assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), state)
    assert next(vae.parameters()).dtype == torch.float32
    assert next(reference.pipe.components["vae"].parameters()).dtype == torch.float16
    assert not vae.use_tiling and vae.tile_sample_min_size == 8


def test_successful_conversion_preserves_real_planner_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    model, vae, latent = fixture()
    planner = memory.Planner(torch, torch.device("cpu"))
    planner.cuda = True  # Install production interception; all tensors/operations stay CPU.
    held = planner.adopt("generation", "vae", vae, None, tiers.HOST, forward_only=True)
    planner.cuda = False
    convert = vae.to
    calls = []

    def counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["dtype"])
        return convert(*args, **kwargs)

    monkeypatch.setattr(vae, "to", counted)
    sdxl.SdxlModel._decode_vae(model, latent, 0)
    epoch = held.conversion_epoch
    assert epoch > 0 and calls == [torch.float32]
    key = memory.OpKey(held.kind, "decode", ())
    observed = memory.Record(ok=4096)
    planner.records[key] = observed
    sdxl.SdxlModel._decode_vae(model, latent, 0)
    assert held.conversion_epoch == epoch and planner.records[key] is observed
    assert calls == [torch.float32]


@pytest.mark.parametrize("previous", [None, torch.bfloat16])
def test_partial_conversion_failure_never_marks_whole_vae_ready(
    monkeypatch: pytest.MonkeyPatch, previous: Any
) -> None:
    model, vae, latent = fixture()
    if previous is not None:
        vae.to(dtype=previous)
        model.pipe._vae_execution_dtype = previous
    convert = vae.to
    attempts = 0

    def fail_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first = next(vae.parameters())
            first.data = first.data.to(dtype=kwargs["dtype"])
            raise torch.OutOfMemoryError("conversion interrupted")
        return convert(*args, **kwargs)

    monkeypatch.setattr(vae, "to", fail_once)
    with pytest.raises(torch.OutOfMemoryError, match="conversion interrupted"):
        sdxl.SdxlModel._decode_vae(model, latent, 0)
    assert next(vae.parameters()).dtype == torch.float32
    assert any(t.dtype == (previous or torch.float16) for t in vae.parameters())
    assert model.pipe._vae_execution_dtype is None
    sdxl.SdxlModel._decode_vae(model, latent, 0)
    assert attempts == 2 and all(t.dtype == torch.float32 for t in vae.parameters())
    assert model.pipe._vae_execution_dtype == torch.float32


def test_failed_decode_keeps_successful_dtype_and_restores_tile_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    model, vae, latent = fixture()
    decode = vae.decode
    failure = torch.OutOfMemoryError("decode interrupted")

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise failure

    monkeypatch.setattr(vae, "decode", fail)
    with pytest.raises(torch.OutOfMemoryError) as caught:
        sdxl.SdxlModel._decode_vae(model, latent, 16)
    assert caught.value is failure
    assert model.pipe._vae_execution_dtype == torch.float32 and not vae.use_tiling
    assert vae.tile_sample_min_size == 8
    monkeypatch.setattr(vae, "decode", decode)
    assert torch.isfinite(sdxl.SdxlModel._decode_vae(model, latent, 0)).all()


def test_no_force_upcast_preserves_original_numerical_mode() -> None:
    model, vae, latent = fixture()
    vae.register_to_config(force_upcast=False)
    reference = deepcopy(model)
    assert torch.equal(
        sdxl.SdxlModel._decode_vae(model, latent, 0), legacy_decode(reference, latent, 0)
    )
    assert model.pipe._vae_execution_dtype is None
    assert next(vae.parameters()).dtype == torch.float16


def test_fixture_weights_have_exact_bf16_roundtrip() -> None:
    # This is a property of these weights, not every possible checkpoint: BF16 near
    # the FP16 overflow boundary is not roundtrip safe. CUDA/image proof is separate.
    _, vae, _ = fixture()
    for tensor in vae.state_dict().values():
        assert torch.equal(tensor.bfloat16(), tensor.bfloat16().half().bfloat16())


class CanonicalStore:
    """Immutable CPU checkpoint bytes; actual tiers own dropping and cast replay."""

    def __init__(self, root: Any) -> None:
        self.values = {
            (id(module), name): tensor.detach().clone()
            for module in root.modules()
            for name, tensor in (*module._parameters.items(), *module._buffers.items())
            if tensor is not None
        }

    def unit_loads(self, name: str, keys: frozenset[str]) -> bool:
        return True

    def transient(self, name: str) -> int:
        return 0

    def drop_unit(self, unit: tiers.Unit) -> None:
        for _, module in unit.owners():
            module.to_empty(device="meta", recurse=False)

    def load_unit(self, name: str, unit: tiers.Unit) -> None:
        for _, module in unit.owners():
            module.to_empty(device="cpu", recurse=False)
        with torch.no_grad():
            for module, slot, _, tensor in unit.slots():
                tensor.copy_(self.values[(id(module), slot)])

    def load(self, name: str) -> object:
        raise AssertionError("unit refill expected")

    def drop(self, name: str) -> None:
        raise AssertionError("unit drop expected")


def test_actual_disk_refill_replays_retained_dtype_after_failed_cast() -> None:
    model, vae, latent = fixture()
    store = CanonicalStore(vae)
    planner = memory.Planner(torch, torch.device("cpu"))
    planner.cuda = True
    held = planner.adopt("generation", "vae", vae, store, tiers.HOST, forward_only=True)
    planner.cuda = False
    expected = sdxl.SdxlModel._decode_vae(model, latent, 0).clone()
    epoch = held.conversion_epoch
    key = memory.OpKey(held.kind, "decode", ())
    observed = memory.Record(ok=4096)
    planner.records[key] = observed
    for unit in held.part.units:
        tiers.to_disk(torch, held.part, unit)
        assert unit.pending and not unit.applied
    unit = held.part.units[-1]
    original = unit.pending[0]
    failed = False

    def cast_once(tensor: Any) -> Any:
        nonlocal failed
        if not tensor.is_meta and not failed:
            failed = True
            raise RuntimeError("refill conversion interrupted")
        return original(tensor)

    unit.pending[0] = tiers.DTypeConvert(cast_once)
    with pytest.raises(RuntimeError, match="refill conversion interrupted"):
        tiers.from_disk(torch, held.part, unit)
    assert unit.tier == tiers.DISK and unit.pending and not unit.applied
    assert model.pipe._vae_execution_dtype == torch.float32
    for unit in held.part.units:
        tiers.from_disk(torch, held.part, unit)
        assert not unit.pending and unit.applied
    assert all(parameter.dtype == torch.float32 for parameter in vae.parameters())
    actual = sdxl.SdxlModel._decode_vae(model, latent, 0)
    assert torch.equal(actual, expected)
    assert held.conversion_epoch == epoch and planner.records[key] is observed
