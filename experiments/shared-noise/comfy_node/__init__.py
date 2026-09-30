# common_ksampler derives from ComfyUI nodes.py, GPL-3.0-or-later; see LICENSE.ComfyUI.
"""Explicit private node; original KSampler remains untouched."""

import comfy.sample
import comfy.utils
import latent_preview
import nodes
from .shared_noise import comfy_noise


def common_ksampler(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent,
    denoise=1.0,
    disable_noise=False,
    start_step=None,
    last_step=None,
    force_full_denoise=False,
):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    if disable_noise:
        noise = comfy.sample.prepare_empty_noise(latent_image)
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=denoise,
        disable_noise=disable_noise,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=force_full_denoise,
        noise_mask=noise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return (out,)


class SharedNoiseKSampler(nodes.KSampler):
    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
    ):
        return _observed_sample(
            self,
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )


NODE_CLASS_MAPPINGS = {"PrivateSharedNoiseKSampler": SharedNoiseKSampler}

# Official keyed wrappers belong only to this private clone/request.
from .observer import Record, CURRENT, first_group, configuration, branch_digests
import inspect
import comfy.patcher_extension


def _sampler(executor, *args, **kwargs):
    rec = CURRENT.get()
    bound = inspect.signature(executor.original).bind(*args, **kwargs).arguments
    rec.add(
        "sampler_schedule",
        sigmas=rec.tensor(bound["sigmas"]),
        supplied_noise=rec.tensor(bound["noise"]),
        initial_latent=rec.tensor(bound["latent_image"]),
    )
    return executor(*args, **kwargs)


def _predict(executor, *args, **kwargs):
    rec = CURRENT.get()
    if rec.group_calls:
        return executor(*args, **kwargs)
    bound = inspect.signature(executor.original).bind(*args, **kwargs).arguments
    rec.add(
        "initial_sampler_state",
        rng=rec.rng(bound["x"].device),
        state=rec.tensor(bound["x"]),
        sigma=rec.tensor(bound["timestep"]),
        quantity="sampler_state_before_BaseModel.calculate_input",
    )
    with first_group(True):
        return executor(*args, **kwargs)


def _network(executor, *args, **kwargs):
    rec = CURRENT.get()
    if not rec.active:
        return executor(*args, **kwargs)
    rec.network_calls += 1
    if rec.network_calls > 2:
        raise ValueError("unsupported private firststep batching")
    bound = inspect.signature(executor.original).bind(*args, **kwargs).arguments
    options = bound.get(
        "transformer_options", bound.get("kwargs", {}).get("transformer_options", {})
    )
    identities = options.get("cond_or_uncond")
    if (
        not isinstance(identities, list)
        or not identities
        or any(type(v) is not int or v not in (0, 1) for v in identities)
    ):
        raise ValueError("unknown conditional branch mapping")
    tensors = {k: v for k, v in bound.items() if hasattr(v, "dtype")}
    values = {k: rec.tensor(v) for k, v in tensors.items()}
    nested = bound.get("kwargs", {})
    tensors.update({k: v for k, v in nested.items() if hasattr(v, "dtype")})
    values.update({k: rec.tensor(v) for k, v in nested.items() if hasattr(v, "dtype")})
    branches = ["positive" if v == 0 else "negative" for v in identities]
    if tensors["x"].shape[0] != len(branches):
        raise ValueError("unsupported condition chunk-to-batch mapping")
    row = rec.add(
        "network",
        complete=False,
        ordinal=rec.network_calls,
        inputs=values,
        branches=branches,
        branch_inputs=branch_digests(rec, branches, tensors),
        output_kind="epsilon" if rec.model == "sdxl" else "flow_velocity",
        model=rec.model,
        training=bool(executor.class_obj.training),
        quantity="after_BaseModel.calculate_input_and_dtype_cast",
    )
    try:
        result = executor(*args, **kwargs)
        row.update(
            output=rec.tensor(result),
            branch_outputs=branch_digests(rec, branches, {"prediction": result}),
            complete=True,
        )
        return result
    except BaseException as exc:
        row["exception"] = type(exc).__name__
        raise


def _observed_sample(
    self,
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    denoise=1.0,
):
    family = {1005: "sdxl", 1006: "anima"}.get(seed)
    rec = Record("comfy", family, seed)
    rec.request = {
        "seed": seed,
        "steps": steps,
        "guidance": cfg,
        "sampler": sampler_name,
        "scheduler": scheduler,
        "denoise": denoise,
    }
    token = CURRENT.set(rec)
    error = None
    try:
        rec.add(
            "execution_config",
            unet=configuration(model.model.model_config.unet_config),
            sampling=configuration(model.model.model_config.sampling_settings),
            sampler=sampler_name,
            scheduler=scheduler,
            steps=steps,
            cfg=cfg,
            prediction_type="epsilon" if family == "sdxl" else "flow_velocity",
        )
        private = model.clone()
        for kind, wrapper in [
            (comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, _sampler),
            (comfy.patcher_extension.WrappersMP.PREDICT_NOISE, _predict),
            (comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _network),
        ]:
            private.add_wrapper_with_key(kind, "private_shared_noise", wrapper)
        return common_ksampler(
            private,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )
    except BaseException as exc:
        error = exc
        raise
    finally:
        CURRENT.reset(token)
        try:
            rec.finish(error)
        except Exception:
            if error is None:
                raise
