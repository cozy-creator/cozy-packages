"""Small byte-exact observations inside an already admitted Runtime component scope.

Expected persistent values name checkpoint 9b743990...a23cd and were compared with
original HF42ed227e carrier bytes. Final AdaLN tables name that published checkpoint;
the VAE rotary expectation comes from its exact Diffusers0.40 constructor/config.
This module never opens a Store or writes weights. It observes CPU tensors too,
so its byte contract can be tested independently of CUDA placement.
"""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple

import torch


class _Sample(NamedTuple):
    key: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    first_rows: int = 0


_SAMPLES: dict[str, tuple[_Sample, ...]] = {
    "fl2va_dit": (
        _Sample(
            "proj_in.weight",
            "f32",
            (5376, 96),
            "26ff2714f21c25539a4f80040735274cc72e3c4a9da82769925df6a1102a301d",
            0,
        ),
        _Sample(
            "proj_out.weight",
            "f32",
            (96, 5376),
            "8c93408b8463d037c658766c82995ec2a401076da5a4d4c245f22ab5ed4c2b97",
            0,
        ),
        _Sample(
            "proj_out.bias",
            "f32",
            (96,),
            "5b15b024d02977f8ddc7e3a901b5a1fa10101430c1abf2e91ea78698c921a38e",
            0,
        ),
        _Sample(
            "norm_out.norm.weight",
            "bf16",
            (5376,),
            "91ac17792929e0a84533b28caf71ac831ec3d666c07a473393639be20392f5ce",
            0,
        ),
        _Sample(
            "norm_out.table",
            "bf16",
            (59, 2, 5376),
            "93b9778c49bd793741a9d55dd6136a42beda0981bad558b6010667d8040c3158",
            0,
        ),
    ),
    "ref2va_dit": (
        _Sample(
            "proj_in.weight",
            "f32",
            (5376, 96),
            "2a6a4f0ee979fac144025127676912fd6211ef69667e5f9892a4fbb84197c147",
            0,
        ),
        _Sample(
            "proj_out.weight",
            "f32",
            (96, 5376),
            "2946e0c2df98704f9bd01510cf9239892fb2e0f8fc267a1f05f20c41e32fc3a8",
            0,
        ),
        _Sample(
            "proj_out.bias",
            "f32",
            (96,),
            "b53aa5bbb89bf59781a91d20005846a9d1884b7750b16c8113310712f31b40b1",
            0,
        ),
        _Sample(
            "norm_out.norm.weight",
            "bf16",
            (5376,),
            "79493dc15653c54965b6305c1dd33b2532a3dde98792de8b45740a18d466ee56",
            0,
        ),
        _Sample(
            "norm_out.table",
            "bf16",
            (59, 2, 5376),
            "a1163bad4fdc2d0bd8038ffef3cbd2d12688e885050907a543cdfa704fe3979f",
            0,
        ),
        _Sample(
            "transformer_blocks.0.attn.to_q.weight",
            "bf16",
            (7168, 5376),
            "c9f5adcdf8b4c6662a6448490c50d5fbb5c0c2827679fcd96ec538f35a0d181c",
            128,
        ),
        _Sample(
            "transformer_blocks.0.attn.to_k.weight",
            "bf16",
            (7168, 5376),
            "8bcfc2e5e67a06d0e05baad1c170b0d69082e2eca92522bf70b25032cf485cbd",
            128,
        ),
        _Sample(
            "transformer_blocks.0.attn.to_v.weight",
            "bf16",
            (7168, 5376),
            "d82e465e48a30006b3954c204241dfae570b2f49b8c1129d93ec421243a4ec34",
            128,
        ),
    ),
    "video_vae": (
        _Sample(
            "encoder.conv_in.weight",
            "f32",
            (128, 3, 3, 3, 3),
            "d519f69e6579cc4a9f301eb65a836e62629ee2d31b3476fc30130552ed150d00",
            0,
        ),
        _Sample(
            "post_quant_conv.weight",
            "f32",
            (24, 24, 1, 1, 1),
            "cbf1865066e0b7b7b582d0b0e80dfad086c20f0b3c39f01ddd5789aa22ecb700",
            0,
        ),
        _Sample(
            "decoder.proj_in.weight",
            "f32",
            (2048, 24),
            "9df43d1b5be0bdfbea5c6ed7a03647ff16ed74b9c74c88f1d92e297088995c16",
            0,
        ),
        _Sample(
            "decoder.norm_out.weight",
            "f32",
            (2048,),
            "2b0ceafd90da467879e2e3815c9c633d32ed34a2de9f4958a8170b8edbd4465e",
            0,
        ),
        _Sample(
            "decoder.proj_out.bias",
            "f32",
            (3072,),
            "5b5aa0ebaeb657238c1df4f3104b1d95d0bcf581ba4add2338413f57756dfecb",
            0,
        ),
        _Sample(
            "decoder.register_tokens",
            "f32",
            (1, 4, 2048),
            "f85ae9f5cd3e3873dbc6573bb21943dde6e04a9a40d82c45d158ba875b38bbf7",
            0,
        ),
        _Sample(
            "decoder.rope.inv_freq",
            "f32",
            (8,),
            "626e97a52a3457af808357a8bc255ca1303cb735abf49fecfe78e28ab4bbb824",
            0,
        ),
    ),
}


def resident_hashes(component: str, module: Any) -> tuple[dict[str, str], dict[str, str]]:
    """Return (actual, expected), keyed ``component/tensor_name``.

    Actual values are SHA256 hex strings over original-dtype contiguous tensor
    bytes, or bounded ``ERROR ...`` strings for a missing, malformed, or parked
    subject. Wrong values produce a different hash; they never abort inference.
    Q/K/V subjects select only the first128rows after checking the full shape.
    Unknown component names raise ValueError as a diagnostic programming error.
    """
    if component not in _SAMPLES:
        raise ValueError(f"unsupported resident sample component: {component}")
    actual: dict[str, str] = {}
    expected: dict[str, str] = {}
    dtypes = {"f32": torch.float32, "bf16": torch.bfloat16}
    for sample in _SAMPLES[component]:
        name = f"{component}/{sample.key}"
        expected[name] = sample.sha256
        try:
            try:
                tensor = module.get_parameter(sample.key)
            except AttributeError:
                tensor = module.get_buffer(sample.key)
        except AttributeError:
            actual[name] = "ERROR missing tensor"
            continue
        if not isinstance(tensor, torch.Tensor):
            actual[name] = "ERROR not a tensor"
            continue
        if tensor.dtype != dtypes[sample.dtype]:
            actual[name] = f"ERROR dtype={tensor.dtype} expected={dtypes[sample.dtype]}"
            continue
        if tuple(tensor.shape) != sample.shape:
            actual[name] = f"ERROR shape={tuple(tensor.shape)} expected={sample.shape}"
            continue
        if tensor.is_meta or not tensor.untyped_storage().nbytes():
            actual[name] = "ERROR unmaterialized tensor"
            continue
        selected = tensor.detach()
        if sample.first_rows:
            selected = selected[: sample.first_rows]
        if selected.numel() * selected.element_size() > 4 << 20:
            actual[name] = "ERROR sample exceeds 4MiB bound"
            continue
        try:
            # Preserve dtype: a float conversion would hash different bytes.
            host = selected.cpu().contiguous().view(torch.uint8).numpy()
            actual[name] = hashlib.sha256(memoryview(host).cast("B")).hexdigest()
        except RuntimeError:
            actual[name] = "ERROR tensor copy failed"
    return actual, expected
