"""CPU tensors and real Diffusers guider/block driver; no model weights or device probes."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers.guiders.classifier_free_guidance import ClassifierFreeGuidance
from diffusers.modular_pipelines.anima.denoise import AnimaLoopDenoiser

from anima.batched_cfg import BatchedAnimaLoopDenoiser, choose_layout


class Transformer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor]:
        self.calls.append(kwargs)
        x = kwargs["hidden_states"]
        # Independent per-example math; distinct conditioning and latent channels.
        bias = kwargs["encoder_hidden_states"].mean(dim=(1, 2))
        return (x * 2 + bias.reshape(-1, 1, 1, 1, 1),)


def state(*, unequal: bool = False) -> Any:
    return SimpleNamespace(
        num_inference_steps=10,
        dtype=torch.float32,
        latent_model_input=torch.arange(16, dtype=torch.float32).reshape(2, 2, 1, 2, 2),
        timestep=torch.tensor([0.4, 0.4]),
        padding_mask=torch.zeros(1, 1, 2, 2),
        prompt_embeds=torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        negative_prompt_embeds=torch.full((2, 4 if unequal else 3, 4), -2.0),
    )


@pytest.mark.parametrize("step", [0, 2, 6, 7, 9])
@pytest.mark.parametrize("rescale", [0.0, 0.3])
@pytest.mark.parametrize("original", [False, True])
def test_batch_mapping_guidance_rescale_and_interval(
    step: int, rescale: float, original: bool
) -> None:
    results = []
    for block in [AnimaLoopDenoiser(), BatchedAnimaLoopDenoiser()]:
        guider = ClassifierFreeGuidance(
            guidance_scale=4.5,
            guidance_rescale=rescale,
            use_original_formulation=original,
            start=0.2,
            stop=0.7,
        )
        transformer = Transformer()
        value = state()
        components: Any = SimpleNamespace(guider=guider, transformer=transformer)
        block(components, value, step, torch.tensor(400))
        results.append((value.noise_pred, transformer.calls))
    torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
    active = 2 <= step < 7
    assert len(results[0][1]) == (2 if active else 1)
    assert len(results[1][1]) == 1
    call = results[1][1][0]
    assert call["hidden_states"].shape[0] == (4 if active else 2)
    assert call["padding_mask"].shape[0] == 1  # Cosmos repeats this itself
    if active:
        torch.testing.assert_close(call["encoder_hidden_states"][:2], state().prompt_embeds)
        torch.testing.assert_close(
            call["encoder_hidden_states"][2:], state().negative_prompt_embeds
        )
        assert torch.equal(call["timestep"], torch.tensor([0.4] * 4))


def test_unknown_guider_and_unequal_context_keep_existing_path() -> None:
    class CustomGuidance(ClassifierFreeGuidance):
        pass

    for guider, unequal in [
        (CustomGuidance(guidance_scale=4.5), False),
        (ClassifierFreeGuidance(guidance_scale=4.5), True),
    ]:
        transformer = Transformer()
        value = state(unequal=unequal)
        BatchedAnimaLoopDenoiser()(
            SimpleNamespace(guider=guider, transformer=transformer), value, 0, torch.tensor(400)
        )
        assert len(transformer.calls) == 2
        assert all(c["hidden_states"].shape[0] == 2 for c in transformer.calls)


def test_fbc_layout_is_selected_before_execution_and_retries() -> None:
    assert choose_layout(0)[0] == "batched"
    assert choose_layout(0.075)[0] == "sequential"
    assert choose_layout(0.075) == choose_layout(0.075)


def test_real_workflow_installs_experiment_on_executed_tree() -> None:
    import anima

    pipeline = anima._text2image_pipeline(torch.device("cpu"), anima._Phases(), batch_cfg=True)
    # Public blocks is a copy; checking its type is okay, but execution below uses
    # the kept block tree to catch the old copy-only instrumentation failure.
    blocks = pipeline._blocks
    block = blocks.sub_blocks["denoise.denoise"].sub_blocks["denoiser"]
    assert isinstance(block, BatchedAnimaLoopDenoiser)
    transformer = Transformer()
    block(
        SimpleNamespace(guider=ClassifierFreeGuidance(guidance_scale=4.5), transformer=transformer),
        state(),
        0,
        torch.tensor(400),
    )
    assert len(transformer.calls) == 1


def test_batch_failure_propagates_without_retrying_sequentially() -> None:
    class FailingTransformer(Transformer):
        def __call__(self, **kwargs: Any) -> tuple[torch.Tensor]:
            self.calls.append(kwargs)
            raise RuntimeError("simulated allocation failure")

    transformer = FailingTransformer()
    guider = ClassifierFreeGuidance(guidance_scale=4.5)
    with pytest.raises(RuntimeError, match="simulated allocation failure"):
        BatchedAnimaLoopDenoiser()(
            SimpleNamespace(guider=guider, transformer=transformer), state(), 0, torch.tensor(400)
        )
    assert len(transformer.calls) == 1
    assert transformer.calls[0]["hidden_states"].shape[0] == 4


def test_guidance_one_remains_single_prediction() -> None:
    transformer = Transformer()
    BatchedAnimaLoopDenoiser()(
        SimpleNamespace(guider=ClassifierFreeGuidance(guidance_scale=1.0), transformer=transformer),
        state(),
        0,
        torch.tensor(400),
    )
    assert len(transformer.calls) == 1
    assert transformer.calls[0]["hidden_states"].shape[0] == 2


def test_actual_layout_provenance_reports_interval_transition_once() -> None:
    reports: list[tuple[str, str]] = []
    block = BatchedAnimaLoopDenoiser(lambda layout, reason: reports.append((layout, reason)))
    components = SimpleNamespace(
        guider=ClassifierFreeGuidance(guidance_scale=4.5, start=0.2, stop=0.7),
        transformer=Transformer(),
    )
    for step in [0, 1, 2, 3, 7]:
        block(components, state(), step, torch.tensor(400))
    assert [layout for layout, _ in reports] == ["single", "batched"]
