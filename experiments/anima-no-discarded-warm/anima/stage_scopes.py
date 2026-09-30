"""Transparent call adapters for the existing, already-selected upstream block tree.

All upstream fields/properties/sub-blocks stay on the original instance. Only its call
enters an existing Model method; the original class implementation executes unchanged.
A copied tree retains the adapter class and therefore the same Model scope owner.
"""

from typing import Any, Callable


def scope_block(block: Any, invoke: Callable[..., Any]) -> None:
    upstream = type(block)

    def call(self: Any, components: Any, state: Any, *args: Any, **kwargs: Any) -> Any:
        return invoke(
            upstream.__call__.__get__(self, type(self)), components, state, *args, **kwargs
        )

    scoped = type(
        f"Scoped{upstream.__name__}", (upstream,), {"__call__": call, "__module__": __name__}
    )
    block.__class__ = scoped


def install_scopes(blocks: Any, model: Any) -> None:
    """A reviewed text2image workflow only; unknown upstream topology is not guessed."""
    expected = (
        "text_encoder",
        "denoise.text_conditioning",
        "denoise.input",
        "denoise.prepare_latents",
        "denoise.set_timesteps",
        "denoise.denoise",
        "decode.decode",
        "decode.postprocess",
    )
    names = tuple(name for name in blocks.sub_blocks if name != "conditioning")
    if names != expected:
        raise RuntimeError(f"Anima text2image scope boundaries changed: {names}")
    scope_block(blocks.sub_blocks["text_encoder"], model.encode_stage)
    scope_block(blocks.sub_blocks["denoise.text_conditioning"], model.condition_stage)
    # input reads transformer.dtype; latent/schedule setup uses its captured value and
    # execution_device. Keep all denoising preparation under the same truthful boundary.
    for name in (
        "denoise.input",
        "denoise.prepare_latents",
        "denoise.set_timesteps",
        "denoise.denoise",
    ):
        scope_block(blocks.sub_blocks[name], model.denoise_stage)
    scope_block(blocks.sub_blocks["decode.decode"], model.decode_stage)
    # Postprocess reads output tensors and image_processor only, never model weights.
