"""Real tiny upstream Anima modules, original block driver and actual Model scopes; CPU only."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen3Config, Qwen3Model, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import anima


def fixture_model() -> anima.AnimaModel:
    torch.manual_seed(37)
    encoder = Qwen3Model(
        Qwen3Config(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    conditioner = AnimaTextConditioner(
        source_dim=16,
        target_dim=16,
        model_dim=16,
        num_layers=1,
        num_attention_heads=2,
        target_vocab_size=8,
        min_sequence_length=4,
    )
    transformer = CosmosTransformer3DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=16,
        num_layers=2,
        mlp_ratio=2,
        text_embed_dim=16,
        adaln_lora_dim=8,
        max_size=(8, 16, 16),
        extra_pos_embed_type=None,
    )
    vae = AutoencoderKLQwenImage(
        base_dim=4,
        z_dim=4,
        dim_mult=[1, 1, 1, 1],
        num_res_blocks=1,
        temperal_downsample=[False, True, True],
        latents_mean=[0] * 4,
        latents_std=[1] * 4,
    )
    vae.enable_tiling(
        tile_sample_min_height=256,
        tile_sample_min_width=256,
        tile_sample_stride_height=192,
        tile_sample_stride_width=192,
    )
    tokenizer = Tokenizer(
        WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "a": 2, "lake": 3, "no": 4, "people": 5, "warm": 6, "": 7},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]"
    )
    pipe = SimpleNamespace(
        components={
            "text_encoder": encoder.eval(),
            "text_conditioner": conditioner.eval(),
            "transformer": transformer.eval(),
            "vae": vae.eval(),
        },
        tokenizer=tokenizer,
        t5_tokenizer=tokenizer,
        scheduler_config=dict(FlowMatchEulerDiscreteScheduler().config),
    )
    return anima.AnimaModel.for_test(pipe=pipe)


def baseline(
    model: anima.AnimaModel, steps: int = 2, threshold: float = 0.0
) -> tuple[Any, torch.Tensor, Any]:
    pipe = model.pipe
    pipeline = anima._text2image_pipeline(torch.device("cpu"), anima._Phases())
    pipeline.register_components(
        **pipe.components,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler_config),
        tokenizer=pipe.tokenizer,
        t5_tokenizer=pipe.t5_tokenizer,
    )
    pipeline.guider.guidance_scale = 4.5
    pipeline.guider._start, pipeline.guider._stop = 0, 1
    generator = torch.Generator(device="cpu").manual_seed(1006)
    restore = anima._apply_first_block_cache(
        pipe.components["transformer"], pipeline.guider, threshold
    )
    try:
        state = pipeline(
            prompt="a lake",
            negative_prompt="no people",
            width=32,
            height=32,
            num_inference_steps=steps,
            generator=generator,
            output_type="pt",
        )
    finally:
        restore()
    return state, generator.get_state(), pipeline


@pytest.mark.parametrize("steps,threshold", [(2, 0.0), (30, 0.0), (2, 0.075)])
def test_real_upstream_pipeline_scoped_matches_state_rng_and_cfg(
    steps: int, threshold: float
) -> None:
    model = fixture_model()
    counts = []
    hook = model.pipe.components["transformer"].register_forward_pre_hook(
        lambda module, args, kwargs: counts.append(kwargs["hidden_states"].shape[0]),
        with_kwargs=True,
    )
    with torch.no_grad():
        before_global_rng = torch.get_rng_state().clone()
        expected, rng, unwrapped = baseline(model, steps, threshold)
        expected_global_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before_global_rng)
        expected_counts = list(counts)
        counts.clear()
        pipeline, generator, restore = model.prepare_request(
            "a lake", "no people", 32, 32, steps, 4.5, (0, 1), threshold, 1006, anima._Phases()
        )
        got = pipeline(
            prompt="a lake",
            negative_prompt="no people",
            width=32,
            height=32,
            num_inference_steps=steps,
            generator=generator,
            output_type="pt",
        )
        restore()
    hook.remove()
    assert expected_counts == counts == [1] * (steps * 2)
    assert torch.equal(rng, generator.get_state())
    assert torch.equal(expected_global_rng, torch.get_rng_state())
    assert expected.values.keys() == got.values.keys()
    for key in expected.values:
        a, b = expected.get(key), got.get(key)
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b), key
    assert torch.equal(expected.get("images"), got.get("images"))
    methods = [call.method for call in model.harness.calls]
    assert methods == [
        "prepare_request",
        "encode_stage",
        "condition_stage",
        "denoise_stage",
        "denoise_stage",
        "denoise_stage",
        "denoise_stage",
        "decode_stage",
    ]
    assert next(c for c in model.harness.calls if c.method == "decode_stage").components == ("vae",)
    assert (
        pipeline.vae.tile_sample_min_height == 256 and pipeline.vae.tile_sample_stride_height == 192
    )
    copied = pipeline.blocks  # actual upstream property DEEPCOPIES the execution tree
    assert type(copied.sub_blocks["text_encoder"]) is type(
        pipeline._blocks.sub_blocks["text_encoder"]
    )
    for key in unwrapped.blocks.sub_blocks:
        original = unwrapped.blocks.sub_blocks[key]
        wrapped = copied.sub_blocks[key]
        for attribute in (
            "inputs",
            "intermediate_outputs",
            "expected_components",
            "expected_configs",
            "outputs",
        ):
            assert getattr(original, attribute) == getattr(wrapped, attribute), attribute
    assert not torch.cuda.is_initialized()


def test_deepcopied_actual_tree_executes_same_scopes_and_releases_prior_anchors() -> None:
    from cozy_runtime.internal import memory
    from cozy_runtime.internal.executor import _Scopes

    model = fixture_model()
    with torch.no_grad():
        expected, expected_rng, _ = baseline(model)
    planner = memory.Planner(torch, torch.device("cpu"))
    object.__setattr__(model, "_cozy_residency", _Scopes(planner, "construction"))
    checks = []
    hooks = []
    for name, module in model.pipe.components.items():

        def check(module: Any, args: Any, kwargs: Any, expected: str = name) -> None:
            active = (
                set().union(*(names for _, names in planner.scopes)) if planner.scopes else set()
            )
            assert expected in active
            if expected == "vae":
                assert active == {"vae"}
            if expected == "transformer":
                assert "text_encoder" not in active and "text_conditioner" not in active
            checks.append((expected, sorted(active)))

        target = module.decoder if name == "vae" else module
        hooks.append(target.register_forward_pre_hook(check, with_kwargs=True))
    try:
        pipeline, generator, restore = model.prepare_request(
            "a lake", "no people", 32, 32, 2, 4.5, (0, 1), 0.0, 1006, anima._Phases()
        )
        assert not planner.scopes
        pipeline._blocks = pipeline.blocks  # execute the actual public-property deepcopy
        with torch.no_grad():
            image = pipeline(
                prompt="a lake",
                negative_prompt="no people",
                width=32,
                height=32,
                num_inference_steps=2,
                generator=generator,
                output="images",
                output_type="pt",
            )
        restore()
        assert image.shape[1] == 3 and not planner.scopes
        assert torch.equal(image, expected.get("images"))
        assert torch.equal(generator.get_state(), expected_rng)
        assert any(name == "vae" for name, _ in checks)
    finally:
        for hook in hooks:
            hook.remove()


def test_external_orchestrator_and_warm_share_identical_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = fixture_model()
    calls = []
    original = anima.render_request

    def record(owner: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append((owner, args))
        # Warm's authored512 request is observed without downsizing it or launching a costly CPU render.
        return None

    monkeypatch.setattr(anima, "render_request", record)
    model.warm(SimpleNamespace(raise_if_cancelled=lambda: None))
    assert len(calls) == 1 and calls[0][0] is model and calls[0][1][2:5] == (512, 512, 1)
    assert not hasattr(type(model), "render")  # no implicit ALL-scoped public orchestrator
    monkeypatch.setattr(anima, "render_request", original)
    with torch.no_grad():
        expected, _, _ = baseline(model)
        actual = anima.render_request(
            model, "a lake", "no people", 32, 32, 2, 4.5, (0, 1), 0.0, 1006, anima._Phases()
        )
    assert torch.equal(actual, expected.get("images"))


def test_unknown_workflow_boundary_refuses_before_rewriting() -> None:
    from anima.stage_scopes import install_scopes

    model = fixture_model()
    pipeline = anima._text2image_pipeline(torch.device("cpu"), anima._Phases())
    blocks = pipeline.blocks
    blocks.sub_blocks.pop("decode.decode")
    old_types = [type(block) for block in blocks.sub_blocks.values()]
    with pytest.raises(RuntimeError, match="scope boundaries changed"):
        install_scopes(blocks, model)
    assert [type(block) for block in blocks.sub_blocks.values()] == old_types


def test_failed_stage_releases_scope_and_restores_original_fbc_binding() -> None:
    from cozy_runtime.internal import memory
    from cozy_runtime.internal.executor import _Scopes
    from cozy_runtime.author import original_forward

    model = fixture_model()
    planner = memory.Planner(torch, torch.device("cpu"))
    object.__setattr__(model, "_cozy_residency", _Scopes(planner, "construction"))
    transformer = model.pipe.components["transformer"]
    before = original_forward(transformer)
    original = ValueError("fixture conditioning failure")

    def fail(module: Any, args: Any) -> None:
        raise original

    hook = model.pipe.components["text_conditioner"].register_forward_pre_hook(fail)
    try:
        with torch.no_grad(), pytest.raises(ValueError) as caught:
            anima.render_request(
                model, "a lake", "no people", 32, 32, 2, 4.5, (0, 1), 0.075, 1006, anima._Phases()
            )
        assert caught.value is original
        assert not planner.scopes
        assert original_forward(transformer) == before
        assert model.harness.calls[-1].method != "finish_request"
    finally:
        hook.remove()


def test_poisoned_vae_failure_preserves_exception_and_restores_fbc_without_admission() -> None:
    from cozy_runtime.internal import memory
    from cozy_runtime.internal.executor import _Scopes
    from cozy_runtime.author import original_forward

    model = fixture_model()
    planner = memory.Planner(torch, torch.device("cpu"))
    object.__setattr__(model, "_cozy_residency", _Scopes(planner, "construction"))
    transformer = model.pipe.components["transformer"]
    before = original_forward(transformer)
    original = memory.MemoryRefusal("device_fault", "injected VAE poison", reusable=False)

    def poison(module: Any, args: Any) -> None:
        planner.unusable = "injected VAE poison"
        raise original

    hook = model.pipe.components["vae"].decoder.register_forward_pre_hook(poison)
    try:
        with torch.no_grad(), pytest.raises(memory.MemoryRefusal) as caught:
            anima.render_request(
                model, "a lake", "no people", 32, 32, 2, 4.5, (0, 1), 0.075, 1006, anima._Phases()
            )
        assert caught.value is original
        assert planner.unusable and not planner.scopes
        assert original_forward(transformer) == before
        # The same planner now refuses any cleanup-style admission; restoration above
        # therefore cannot have depended on opening another component scope.
        with pytest.raises(memory.MemoryRefusal):
            planner.enter("construction", "would-mask-original", ("transformer",))
    finally:
        hook.remove()
