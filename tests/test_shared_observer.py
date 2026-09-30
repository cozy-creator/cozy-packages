import contextlib
import copy
import functools
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from cozy_runtime.author import pure
from cozy_runtime.internal import memory, tiers

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "observer", ROOT / "experiments/shared-noise/observer.py"
)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)


def test_real_tiny_sdxl_managed_forward_observation_is_value_rng_and_exception_neutral():
    from diffusers import UNet2DConditionModel

    cfg = dict(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        cross_attention_dim=8,
        norm_num_groups=4,
    )
    torch.manual_seed(37)
    baseline = UNet2DConditionModel(**cfg).eval()

    class Observed(UNet2DConditionModel):
        forward = o.observe_network(UNet2DConditionModel.forward, "sdxl")

    net = pure(Observed(**cfg).eval())
    net.load_state_dict(baseline.state_dict())
    owner = memory.Planner(torch, torch.device("cpu"))
    held = owner.adopt("c", "unet", net, None, tiers.HOST)
    tiers.wrap(net, functools.partial(owner.run, held))
    x = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 4, 8)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = baseline(x, torch.tensor(1), encoder_hidden_states=context).sample
    rec = o.Record("cozy", "sdxl", 1005)
    token = o.CURRENT.set(rec)
    try:
        with torch.no_grad(), o.first_group(True):
            actual = net(x, torch.tensor(1), encoder_hidden_states=context).sample
    finally:
        o.CURRENT.reset(token)
    assert torch.equal(actual, expected) and torch.equal(before, torch.get_rng_state())
    assert len(rec.events) == 1 and rec.events[0]["complete"]
    assert rec.events[0]["branches"] == ["negative", "positive"] and not owner.stack
    error = ValueError("original")

    def fail(self, sample):
        raise error

    wrapped = o.observe_network(fail, "sdxl")
    rec = o.Record("cozy", "sdxl", 1005)
    token = o.CURRENT.set(rec)
    try:
        with pytest.raises(ValueError) as caught:
            with o.first_group(True):
                wrapped(types.SimpleNamespace(config={}, training=False), x)
        assert caught.value is error and not rec.active and rec.events[0]["complete"] is False
    finally:
        o.CURRENT.reset(token)


def test_anima_actual_upstream_loop_and_model_scopes_record_branches_without_rewrite(monkeypatch):
    directory = ROOT / "experiments/anima-shared-noise"
    monkeypatch.syspath_prepend(str(directory))
    import anima

    fixture_spec = importlib.util.spec_from_file_location(
        "stage_fixture", directory / "tests/test_stage_scopes.py"
    )
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    model = fixture.fixture_model()
    original = model.pipe.components["transformer"]
    observed = anima.ObservedCosmos.from_config(original.config)
    observed.load_state_dict(original.state_dict())
    observed.eval()
    model.pipe.components["transformer"] = observed
    obs = sys.modules["anima.observer"]
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        expected, expected_rng, _ = fixture.baseline(model, 2, 0)
    torch.set_rng_state(before)
    rec = obs.Record("cozy", "anima", 1006)
    token = obs.CURRENT.set(rec)
    try:
        with torch.no_grad():
            pipeline, generator, restore = model.prepare_request(
                "a lake", "no people", 32, 32, 2, 4.5, (0, 1), 0, 1006, anima._Phases()
            )
            actual = pipeline(
                prompt="a lake",
                negative_prompt="no people",
                width=32,
                height=32,
                num_inference_steps=2,
                generator=generator,
                output_type="pt",
            )
            restore()
    finally:
        obs.CURRENT.reset(token)
    assert torch.equal(actual.get("images"), expected.get("images")) and torch.equal(
        generator.get_state(), expected_rng
    )
    assert torch.equal(before, torch.get_rng_state())
    networks = [r for r in rec.events if r["event"] == "network"]
    assert len(networks) == 2 and all(r["complete"] for r in networks)
    assert sorted(r["branches"] for r in networks) == [["negative"], ["positive"]]
    assert rec.group_calls == 1 and not rec.active
    assert "denoise_stage" in [c.method for c in model.harness.calls]


def test_digest_bounds_rawbits_and_failed_group():
    rec = o.Record("cozy", "sdxl", 1005)
    bits = torch.tensor([0, -32768, 32257, 32258], dtype=torch.int16)
    v = bits.view(torch.float16)
    assert (
        rec.tensor(v)["sha256"]
        == __import__("hashlib").sha256(bits.view(torch.uint8).numpy().tobytes()).hexdigest()
    )
    with pytest.raises(ValueError):
        rec.tensor(torch.empty(o.MAX_TENSOR_BYTES + 1, dtype=torch.uint8))
    token = o.CURRENT.set(rec)
    try:
        with o.first_group(True):
            pass
        with pytest.raises(ValueError):
            with o.first_group(True):
                pass
        assert not rec.active
    finally:
        o.CURRENT.reset(token)


def test_actual_comfy_wrapper_executor_calls_once_and_preserves_failure():
    import ast

    extension_path = Path(
        "/home/fidika/cozy/.worktrees/ComfyUI/memory-benchmark-20260929/comfy/patcher_extension.py"
    )
    spec = importlib.util.spec_from_file_location("private_patcher_extension", extension_path)
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    source = ROOT / "experiments/shared-noise/comfy_node/__init__.py"
    tree = ast.parse(source.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ("_sampler", "_predict", "_network")
    ]
    ns = {
        "CURRENT": o.CURRENT,
        "first_group": o.first_group,
        "inspect": __import__("inspect"),
        "branch_digests": o.branch_digests,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), ns)

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 1)
            self.calls = 0

        def forward(self, x, timesteps, context, transformer_options):
            self.calls += 1
            return self.conv(x)

    net = Network()
    x = torch.randn(2, 4, 4, 4)
    context = torch.ones(2, 4, 8)
    options = {"cond_or_uncond": [0, 1]}

    class Guider:
        def predict_noise(self, x, timestep, model_options=None, seed=None):
            executor = ext.WrapperExecutor.new_class_executor(net.forward, net, [ns["_network"]])
            return executor.execute(x, timestep, context, options)

    guider = Guider()
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        expected = net.forward(x, torch.ones(2), context, options)
    rec = o.Record("comfy", "sdxl", 1005)
    token = o.CURRENT.set(rec)
    try:
        with torch.no_grad():
            actual = ext.WrapperExecutor.new_class_executor(
                guider.predict_noise, guider, [ns["_predict"]]
            ).execute(x, torch.ones(2), {}, 1005)
        assert (
            torch.equal(actual, expected)
            and net.calls == 2
            and torch.equal(before, torch.get_rng_state())
        )
        assert rec.events[-1]["branches"] == ["positive", "negative"] and rec.events[-1]["complete"]
        assert rec.events[0]["quantity"] == "sampler_state_before_BaseModel.calculate_input"
        assert rec.events[-1]["quantity"] == "after_BaseModel.calculate_input_and_dtype_cast"
    finally:
        o.CURRENT.reset(token)
    error = ValueError("real error")

    def failing(x, timesteps, context, transformer_options):
        raise error

    rec = o.Record("comfy", "sdxl", 1005)
    token = o.CURRENT.set(rec)
    try:
        with pytest.raises(ValueError) as caught:
            with o.first_group(True):
                ext.WrapperExecutor.new_class_executor(failing, net, [ns["_network"]]).execute(
                    x, torch.ones(2), context, options
                )
        assert caught.value is error and not rec.active and rec.events[-1]["complete"] is False
    finally:
        o.CURRENT.reset(token)


def test_full_meta_sdxl_and_cosmos_partition_stays_identical():
    from diffusers import UNet2DConditionModel, CosmosTransformer3DModel
    import json

    configs = [
        (
            "sdxl",
            UNet2DConditionModel,
            Path(
                "/home/fidika/.cozy/outputs/comfy-parity-20260925/assets/sdxl-native/unet/config.json"
            ),
        ),
        (
            "anima",
            CosmosTransformer3DModel,
            ROOT / "experiments/anima-shared-noise/configs/transformer.json",
        ),
    ]
    for family, base, path in configs:
        cfg = json.loads(path.read_text())
        with torch.device("meta"):
            plain = base.from_config(cfg)
            observed = type(
                "PrivateObserved" + base.__name__,
                (base,),
                {"forward": o.observe_network(base.forward, family)},
            ).from_config(cfg)

        def census(net):
            part = tiers.partition("model", net, tiers.DISK)
            return [
                (
                    u.path,
                    u.nbytes,
                    [
                        (p, n, k, tuple(t.shape), str(t.dtype), tuple(t.stride()))
                        for p, m in u.owners()
                        for k, slots in [("p", m._parameters), ("b", m._buffers)]
                        for n, t in slots.items()
                        if t is not None
                    ],
                )
                for u in part.units
            ]

        assert census(plain) == census(observed)
        assert all(t.is_meta for t in plain.parameters()) and not torch.cuda.is_initialized()


def test_equal_anima_contexts_cannot_fabricate_branch_identity():
    rec = o.Record("cozy", "anima", 1006)
    value = torch.ones(1, 2, 3)
    rec.branches = {"positive": rec.tensor(value), "negative": rec.tensor(value)}

    def original(self, hidden_states, encoder_hidden_states):
        pytest.fail("ambiguous branch must refuse before model call")

    wrapped = o.observe_network(original, "anima")
    token = o.CURRENT.set(rec)
    try:
        with pytest.raises(ValueError, match="ambiguous"):
            with o.first_group(True):
                wrapped(
                    types.SimpleNamespace(config={}, training=False), torch.ones(1, 4, 2, 2), value
                )
        assert not rec.active
    finally:
        o.CURRENT.reset(token)


def test_cuda_strided_refuses_before_copy_queries_or_accounting(monkeypatch):
    class CudaFacade(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

        def detach(self):
            pytest.fail("unsupported CUDA layout reached detach/copy")

        def to(self, *args, **kwargs):
            pytest.fail("unsupported CUDA layout reached to")

    source = torch.ones(2, 3).T
    value = torch.Tensor._make_subclass(CudaFacade, source, False)
    assert not value.is_contiguous() and value.device.type == "cuda"
    record = o.Record("cozy", "sdxl", 1005)
    monkeypatch.setattr(o, "allocator", lambda *_: pytest.fail("allocator query before rejection"))
    with pytest.raises(ValueError, match="noncontiguous CUDA"):
        record.tensor(value)
    assert record.copied == record.copy_index == 0
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kind", ["conjugate", "negative"])
def test_lazy_views_refuse_before_copy_even_when_cpu_contiguous(kind, monkeypatch):
    value = (
        torch.ones(3, dtype=torch.complex64).conj()
        if kind == "conjugate"
        else torch._neg_view(torch.ones(3))
    )
    record = o.Record("cozy", "sdxl", 1005)
    monkeypatch.setattr(
        torch.Tensor, "to", lambda *_args, **_kwargs: pytest.fail("lazy view reached copy")
    )
    monkeypatch.setattr(o, "allocator", lambda *_: pytest.fail("lazy view reached allocator query"))
    with pytest.raises(ValueError, match="lazy conjugate/negative"):
        record.tensor(value)
    assert record.copied == record.copy_index == 0
    assert not torch.cuda.is_initialized()


def test_cpu_strided_packing_remains_bounded_and_exact():
    source = torch.arange(24, dtype=torch.float32).reshape(4, 6)[1:, ::2].T
    record = o.Record("cozy", "sdxl", 1005)
    actual = record.tensor(source)
    expected = (
        __import__("hashlib")
        .sha256(source.contiguous().view(torch.uint8).numpy().tobytes())
        .hexdigest()
    )
    assert actual["sha256"] == expected
    assert (
        actual["stride"] == list(source.stride())
        and actual["storage_offset"] == source.storage_offset()
    )
    assert record.copied == source.numel() * source.element_size()
    assert not torch.cuda.is_initialized()
