"""se-008 — the SDXL launch endpoint: `generate`, text to image, four components.

The second launch family, and the ENGINELESS-PATH REFERENCE (`cozy-runtime-sdxl.md`): one
`Model` class with author-written `load`, three public methods each declaring the
heavyweight components it may touch, one entrypoint. There is no engine class, no device
call, no placement, no offload, no pinning and no backend API — there is no such surface on
`cozy_runtime.author` for this file to reach. Whether the UNet was resident when `denoise`
was entered is the runtime's question, decided before entry, and this file cannot observe
the answer.

What it adds over cr-008b's corpus fixture, which it is otherwise faithful to (its banked
1024px pixel digest reproduces here, byte for byte):

  * ASPECT BUCKETS. SDXL fine-tunes are trained on fixed resolutions, not free geometry, so
    the request names a bucket and `Shape(pixels=...)` is how the demand axes are derived
    from a preset the runtime cannot otherwise measure (§1.2). A bucket this endpoint does
    not offer is unspellable, not rounded.
  * `ModelDefault` steps/guidance, and the TWO-LAYER CLAMP. Layer 1 is the field's own
    `Meta` bound, which REJECTS what the caller actually sent. Layer 2 is a deployment's
    visible `Clamp`, which lowers it and says so in the adjustments envelope. A silent
    clamp is the failure both layers exist to prevent.
  * VALUE-PLANE CFG GATING. `guidance <= 1.0` skips the negative branch entirely — the
    turbo/lightning execution shape follows from the RESOLVED VALUE alone, with zero
    adapter awareness in this file. When the adapter runtime (cr-010) lands, a turbo
    recipe supplies that value as an omitted-field default and not one line here changes.
  * AN OUTPUT-INTEGRITY FLOOR. A decode that produced NaNs, or a flat field with no
    picture in it, is a failure this endpoint reports rather than a PNG it publishes.

Code states CAPABILITY; bindings state SELECTION. Nothing here names a repo, release,
checkpoint or revision — `endpoint.toml` and the deploy binding do.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import msgspec
from cozy_runtime.author import (
    App,
    Context,
    ImageAsset,
    ImageFrame,
    Loader,
    Model,
    ModelDefault,
    OutputError,
    Outputs,
    Shape,
    Telemetry,
    uses_components,
)

app = App()


class AspectRatio(Enum):
    """The offered buckets. A ratio absent here does not round — it refuses at decode."""

    SQUARE = "1:1"
    LANDSCAPE = "4:3"
    WIDE = "16:9"
    PORTRAIT = "3:4"
    TALL = "9:16"


#: bucket -> (width, height). SDXL's own training resolutions, each ~1 megapixel and each a
#: multiple of 64 so the latent grid is exact.
_BUCKETS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.SQUARE: (1024, 1024),
    AspectRatio.LANDSCAPE: (1152, 896),
    AspectRatio.WIDE: (1344, 768),
    AspectRatio.PORTRAIT: (896, 1152),
    AspectRatio.TALL: (768, 1344),
}

#: The boot warm pass's geometry and step count. `ctx.boot_warmup` is honoured because the
#: alternative was measured (cl-003, decisions #392): a warm pass that decodes 1024px with
#: every component resident OOMs this card at every boot, so the binding comes up DEGRADED
#: for nothing and the worker serves anyway — a degradation with no cause, on every worker,
#: forever.
_WARM_SIDE = 512
_WARM_STEPS = 1

#: The tokenizer vocabularies this endpoint BUNDLES. They are its own asset, exactly like
#: the model library it imports — not an artifact identifier, not a catalog ref, and not
#: something a construction config may carry (a path in a construction config refuses).
_TOKENIZERS = Path(__file__).resolve().parent


class Txt2ImgInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str = "a photograph of an astronaut riding a horse"
    negative_prompt: str = ""
    aspect_ratio: Annotated[AspectRatio, Shape(pixels=_BUCKETS)] = AspectRatio.SQUARE
    #: LAYER 1 of the two-layer clamp. These bounds REJECT; they never quietly clip. The
    #: `ModelDefault` marker is what makes the field omittable on the wire and concrete
    #: before the handler runs — the handler sees `int`, never `int | None`.
    steps: Annotated[ModelDefault[int], msgspec.Meta(ge=1, le=50)] = 20
    guidance: Annotated[ModelDefault[float], msgspec.Meta(ge=0.0, le=20.0)] = 5.0
    seed: int = 1005


class ImageOutput(msgspec.Struct):
    image: ImageAsset
    width: int
    height: int
    steps: int
    guidance: float
    classifier_free: bool
    """Whether the negative branch ran. Value-plane gating, made observable: a caller can
    see that `guidance <= 1.0` bought it a one-pass step rather than having to trust it."""
    digest: str
    """sha256 of the decoded RGB pixel bytes — the determinism fence over the WHOLE loop,
    not over one step."""


# ------------------------------------------------------------------ the model


class SdxlPipeline:
    """The four constructed component roots, built from the artifact's immutable config.

        text_encoder     196 destinations   0.229 GiB   CLIPTextModel (ViT-L text tower)
        text_encoder_2   517 destinations   1.294 GiB   CLIPTextModelWithProjection (bigG)
        unet            1680 destinations   4.782 GiB   UNet2DConditionModel
        vae              248 destinations   0.156 GiB   AutoencoderKL
                        ────────────────────────────
                        2641 destinations   6.461 GiB

    fp16 everywhere: it is the checkpoint's stored dtype AND the destination's compute
    contract, so a `plain` fill stays an encoding-contract copy. The fp8 rung's UNet
    decodes to the same destinations at fill (cr-006/cr-008c) and this class cannot tell.
    """

    def __init__(self, config: Any) -> None:
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

        mapping = config.mapping()
        self.components: dict[str, Any] = {
            "text_encoder": CLIPTextModel(CLIPTextConfig(**mapping["text_encoder"])).to(
                torch.float16
            ),
            "text_encoder_2": CLIPTextModelWithProjection(
                CLIPTextConfig(**mapping["text_encoder_2"])
            ).to(torch.float16),
            "unet": UNet2DConditionModel.from_config(mapping["unet"]).to(torch.float16),
            "vae": AutoencoderKL.from_config(mapping["vae"]).to(torch.float16),
        }
        self.scheduler_config: dict[str, Any] = dict(mapping["scheduler"])
        self.vae_scale: float = float(mapping["vae"]["scaling_factor"])


def build_pipeline(config: Any) -> SdxlPipeline:
    return SdxlPipeline(config)


class SdxlModel(Model[SdxlPipeline]):
    """Admits by pure topology satisfaction: no structural twins, so no stamp keyword.

    THREE component sets rather than one coarse whole-pipeline set. The coarse form is
    legal and simpler, and on this card it is not servable: 6.461 GiB of weights declared
    resident at once beside a 1024px VAE decode does not fit 8 GiB, and the answer to that
    is the runtime's staging ladder — which has nothing to stage if every method declares
    everything.
    """

    pipe: SdxlPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(SdxlPipeline, factory=build_pipeline)

    @uses_components("text_encoder", "text_encoder_2")
    def encode(self, ids: Any, ids_2: Any) -> tuple[Any, Any]:
        """SDXL's two-tower conditioning. ONE method, BOTH encoders — a composite operation
        declares its components together rather than opening two scopes.

        IT PLACES ITS OWN INPUTS (#534c). Token ids arrive on the host, and the only code
        entitled to say where they go is code that holds a lease on the component they are
        going to: it follows the encoder's own weights to wherever the runtime already put
        them. The caller used to do this with a hard-coded `cuda:0`, which was an author
        naming a device — the one live instance se-001's record left open — and it made this
        endpoint unservable on any envelope the runtime did not place at ordinal zero.
        """
        import torch

        first_encoder = self.pipe.components["text_encoder"]
        second_encoder = self.pipe.components["text_encoder_2"]
        with torch.inference_mode():
            ids = ids.to(first_encoder.get_input_embeddings().weight.device)
            ids_2 = ids_2.to(second_encoder.get_input_embeddings().weight.device)
            first = first_encoder(ids, output_hidden_states=True)
            second = second_encoder(ids_2, output_hidden_states=True)
            prompt = torch.cat([first.hidden_states[-2], second.hidden_states[-2]], dim=-1)
            return prompt, second.text_embeds

    @uses_components("unet")
    def denoise(
        self, latents: Any, timestep: Any, prompt: Any, text_embeds: Any, time_ids: Any
    ) -> Any:
        import torch

        with torch.inference_mode():
            return self.pipe.components["unet"](
                latents,
                timestep,
                encoder_hidden_states=prompt,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            ).sample

    @uses_components("vae")
    def decode(self, latents: Any) -> Any:
        import torch

        with torch.inference_mode():
            return self.pipe.components["vae"].decode(latents / self.pipe.vae_scale).sample


# ------------------------------------------------------------------ the handler


def _finite(torch: Any, value: Any) -> float:
    """The tensor's largest magnitude, with NaN neutralized so the metric is spellable.

    A metric that cannot be serialized is a metric that is not there: the observation emit
    boundary refuses a non-finite value, so the NaN FRACTION is its own number and this one
    stays a real magnitude.
    """
    return round(float(torch.nan_to_num(value, 0.0, 0.0, 0.0).abs().max()), 4)


def _tokenizer(name: str) -> Any:
    """One bundled CLIP tokenizer, built from its own two files.

    Deliberately NOT `from_pretrained`: that spelling takes a string it will resolve
    against the Hub when it is not a directory, so it is a fetch this endpoint might one
    day make by accident — and `fence.py::no-identifiers-in-code` refuses it for exactly
    that reason. The direct constructor takes the two files and cannot reach anywhere.
    """
    import json

    from transformers import CLIPTokenizer

    root = _TOKENIZERS / name
    settings = json.loads((root / "tokenizer_config.json").read_text())
    return CLIPTokenizer(
        vocab_file=str(root / "vocab.json"),
        merges_file=str(root / "merges.txt"),
        errors=settings["errors"],
        pad_token=settings["pad_token"],
        model_max_length=settings["model_max_length"],
    )


def _tokenize(tokenizers: tuple[Any, Any], prompt: str) -> tuple[Any, Any]:
    first, second = tokenizers
    return _ids(first, prompt), _ids(second, prompt)


def _ids(tok: Any, prompt: str) -> Any:
    return tok(
        prompt,
        padding="max_length",
        max_length=tok.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


#: THE OUTPUT-INTEGRITY FLOOR. Two properties of a decoded image, each with a failure this
#: family actually produces:
#:
#:   * a NaN anywhere means the denoise diverged or a component filled wrong, and the PNG
#:     encoder would happily turn it into a grey rectangle;
#:   * a SPREAD at or below the floor means a flat field — the black image a wrong VAE
#:     scaling factor or an unfilled decoder produces, which encodes and looks like a
#:     result until someone opens it.
#:
#: The floor is deliberately far below any real render and far above a flat field, so it
#: fires on the failure and never on a legitimately dark picture.
#:
#: THE TWO CHECKS READ DIFFERENT TENSORS, and that is the whole of what a red arm taught
#: here: the NaN check ran on the QUANTIZED pixels in the first cut, where `torch.isnan`
#: is constant False because a uint8 cannot be NaN — the `clamp(0,1).to(uint8)` that makes
#: an image encodable is exactly what erases the evidence of a diverged decode. So NaN is
#: read off the float decode BEFORE quantization, and spread off the pixels a caller will
#: actually open.
_MIN_SPREAD = 4.0


def _integrity(torch: Any, image: Any, pixels: Any, tel: Telemetry) -> None:
    nan_fraction = float(torch.isnan(image).float().mean())
    tel.metric("image_nan_fraction", round(nan_fraction, 6))
    if nan_fraction > 0.0:
        raise OutputError(
            f"the decode produced NaN over {nan_fraction:.4%} of the image and this "
            "endpoint does not publish it: a non-finite decode is a failed generation, "
            "not a picture with artefacts",
            code="output_integrity_nan",
        )
    spread = float(pixels.to(torch.float32).std())
    tel.metric("image_spread", round(spread, 4))
    if spread <= _MIN_SPREAD:
        raise OutputError(
            f"the decoded image is a flat field (std {spread:.3f} <= {_MIN_SPREAD}): the "
            "generation produced no picture, and an encodable rectangle is not a result",
            code="output_integrity_flat",
        )


@app.entrypoint
def generate(
    ctx: Context,
    payload: Txt2ImgInput,
    model: SdxlModel,
    out: Outputs,
    tel: Telemetry,
) -> ImageOutput:
    """One text-to-image generation: tokenize, encode, denoise, decode, encode a PNG."""
    import torch
    from diffusers import EulerDiscreteScheduler

    view = model.for_request(ctx, seed=payload.seed)
    width, height = _BUCKETS[payload.aspect_ratio]
    steps = payload.steps
    if ctx.boot_warmup:
        width = height = _WARM_SIDE
        steps = _WARM_STEPS
    # The value plane, and the only branch in this file that reads a request number: above
    # 1.0 the negative branch is worth its second forward pass, at or below it is not.
    classifier_free = payload.guidance > 1.0

    with tel.stage("tokenize"):
        tokenizers = (_tokenizer("tokenizer"), _tokenizer("tokenizer_2"))
        ids, ids_2 = _tokenize(tokenizers, payload.prompt)
        if classifier_free:
            neg, neg_2 = _tokenize(tokenizers, payload.negative_prompt)

    with tel.stage("encode"):
        prompt, pooled = model.encode(ids, ids_2)
        if classifier_free:
            negative, neg_pooled = model.encode(neg, neg_2)
    # THE DEVICE ENVELOPE, READ RATHER THAN NAMED (#534c). The conditioning came back from
    # the encoders the runtime placed, so it already carries where this request runs; the
    # latents and the schedule follow it. This is data flow across the boundary an author
    # owns, and it is the whole of what the deleted `torch.device("cuda", 0)` was doing.
    device = prompt.device
    tel.metric("prompt_absmax", _finite(torch, prompt))
    tel.metric("pooled_absmax", _finite(torch, pooled))

    scheduler = EulerDiscreteScheduler.from_config(model.pipe.scheduler_config)
    scheduler.set_timesteps(steps, device=device)
    generator = torch.Generator(device=device).manual_seed(view._seed)
    latents = (
        torch.randn(
            1, 4, height // 8, width // 8, generator=generator, device=device,
            dtype=torch.float16,
        )
        * scheduler.init_noise_sigma
    )
    # SDXL's micro-conditioning: (original_h, original_w, crop_top, crop_left, target_h,
    # target_w). The bucket IS the original size — these fine-tunes were trained on it.
    time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]], device=device, dtype=torch.float16
    )
    if classifier_free:
        batch_prompt = torch.cat([negative, prompt])
        batch_pooled = torch.cat([neg_pooled, pooled])
        batch_ids = torch.cat([time_ids, time_ids])
    else:
        batch_prompt, batch_pooled, batch_ids = prompt, pooled, time_ids

    on_step = tel.step_callback(steps, stage="denoise")
    with tel.stage("denoise"):
        for index, timestep in enumerate(scheduler.timesteps):
            ctx.raise_if_cancelled()
            batch = torch.cat([latents] * 2) if classifier_free else latents
            model_input = scheduler.scale_model_input(batch, timestep)
            noise = model.denoise(model_input, timestep, batch_prompt, batch_pooled, batch_ids)
            if classifier_free:
                uncond, cond = noise.chunk(2)
                noise = uncond + payload.guidance * (cond - uncond)
            if index == 0:
                tel.metric("noise_absmax_step0", _finite(torch, noise))
            latents = scheduler.step(noise, timestep, latents).prev_sample
            on_step(index)
    tel.metric("latent_absmax", _finite(torch, latents))

    with tel.stage("decode"):
        image = model.decode(latents)
    tel.metric("image_absmax", _finite(torch, image))
    pixels = ((image / 2 + 0.5).clamp(0, 1)[0] * 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    _integrity(torch, image, pixels, tel)

    rgb = bytes(pixels.cpu().numpy().tobytes())
    decoded_h, decoded_w = int(pixels.shape[0]), int(pixels.shape[1])
    tel.metric("decoded_pixels", float(decoded_w * decoded_h))
    with tel.stage("encode_png"):
        asset = out.save_image(ImageFrame(decoded_w, decoded_h, rgb), format="png")
    return ImageOutput(
        image=asset,
        width=decoded_w,
        height=decoded_h,
        steps=steps,
        guidance=payload.guidance,
        classifier_free=classifier_free,
        digest=hashlib.sha256(rgb).hexdigest(),
    )
