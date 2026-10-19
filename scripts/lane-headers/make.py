#!/usr/bin/env python3
"""Regenerate the real cozytensors/1 headers `scripts/lane-plans-proof.py` resolves lane plans against.

Each header is what `tensorfs.Store.import_` writes from a sparse copy of a real source (its
safetensors header with a hole for the payload, so only geometry is real):
- upstream single files and folders, through TensorFS's converter tables (banked headers);
- a production lane's v1 header, as a Diffusers folder under D4's names (`tfs dev migrate`'s).

    python make.py <cozy-creator>/crates/tensorfs/vectors <tfs dev headers dir> <out dir> [name...]
"""

import json
import shutil
import struct
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import tensorfs

SEEDVR = "https://raw.githubusercontent.com/ByteDance-Seed/SeedVR/e4de8c24441a67e1b7df56abea10645059bb1185"
#: Upstream sources: banked header (or URL) -> path in the source.
UPSTREAM = {
    "sdxl": {"civitai-sdxl-headers/128078-92696.safetensors.header": "sdxl.safetensors"},
    "anima": {
        "anima-civitai-headers/2945208-2824391.safetensors.header": "anima_baseV10.safetensors",
        "anima-civitai-headers/2945208-2824387.safetensors.header": "anima_baseV10_txt.safetensors",
        "anima-civitai-headers/2945208-2004692.safetensors.header": "qwen_image_vae.safetensors",
    },
    "seedvr2": {
        "seedvr2-headers/seedvr2_ema_7b_fp16.safetensors.header": "seedvr2_ema_7b_fp16.safetensors",
        "seedvr2-headers/ema_vae_fp16.safetensors.header": "ema_vae_fp16.safetensors",
        f"{SEEDVR}/pos_emb.pt": "pos_emb.pt",
        f"{SEEDVR}/neg_emb.pt": "neg_emb.pt",
    },
    "qwen-image-2": {
        f"qwen-image21-headers/{c}/{f}": f"{c}/{f.removesuffix('.header')}"
        for c, files in {
            "transformer": [f"diffusion_pytorch_model-0000{i}-of-00002.safetensors.header" for i in (1, 2)]
            + ["diffusion_pytorch_model.safetensors.index.json"],
            "text_encoder": [f"model-0000{i}-of-00004.safetensors.header" for i in (1, 2, 3, 4)]
            + ["model.safetensors.index.json"],
            "vae": ["diffusion_pytorch_model.safetensors.header"],
        }.items()
        for f in files
    },
}
#: Production lanes (`tfs dev headers`): v1 manifest digest, and D4's component names.
H3 = {"fl2va_dit": "transformer", "ref2va_dit": "transformer_ref", "video_vae": "vae"}
LANES = {
    "flux-3-klein": ("30615fedd4da5f7fa1a406320ff24783ae6ec2fe0e697e472c4a0c3d79e6647f", {}),
    "flux-3-klein-upsampler": ("843b3eba3baca9e3da100194d9a7a185b850218368e19f31b5e3f1795b5c67cb", {}),
    "qwen3.5-9b": ("bd709fa5b03cc36b935e8b32879ad4be2d514657b07f917d285aa5a95369261a", {}),
    "qwen3.5-9b-mtp": ("3c5527a98f98861311fd0cddf10a1daf6fddc75feaaa5c1787dfa9d7e3d0e98d", {}),
    "birefnet": ("e1e415b9d674ebb987c66755032fe2ac10f877d9905dc62983b00c357b4da132", {}),
    "minimax-h3": ("a34a65433b89f6a38eaa483b789919d6a8586731d1974e9437a71db4b8ed5e92", H3),
    "minimax-h3-full": ("f1e86cb6b4935fbf9e79d90ee87366b6d13e2280b1846b4e9bfa0a00242793eb", H3),
}
SOURCES = Path("/tmp/lane-headers")
V1_DTYPES = ["F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I64", "I32", "I16", "I8", "U8", "BOOL"]
WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I64": 8, "I32": 4,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def sparse(path: Path, header: bytes, payload: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header)
        f.truncate(len(header) + payload)


def banked(src: bytes) -> tuple[bytes, int]:
    n = struct.unpack("<Q", src[:8])[0]
    doc = json.loads(src[8 : 8 + n])
    return src[: 8 + n], max(v["data_offsets"][1] for k, v in doc.items() if k != "__metadata__")


class V1:
    """A v1 header's configs and tensors (`[format, configs, assets, encodings, components]`)."""

    def __init__(self, raw: bytes) -> None:
        self.b, self.i = raw, 0
        root = self.value()
        self.configs = dict(root[1])
        self.components = [(c, [(t[0], V1_DTYPES[t[1]], t[2]) for t in ts]) for c, ts in root[4]]

    def value(self) -> Any:
        head = self.b[self.i]
        self.i += 1
        major, info = head >> 5, head & 31
        if info >= 24:
            size = 1 << (info - 24)
            info = int.from_bytes(self.b[self.i : self.i + size], "big")
            self.i += size
        if major == 0:
            return info
        if major in (2, 3):
            out = self.b[self.i : self.i + info]
            self.i += info
            return out.decode() if major == 3 else out
        if major == 4:
            return [self.value() for _ in range(info)]
        raise ValueError(f"CBOR major type {major}")


def from_lane(src: Path, headers: Path, manifest: str, names: dict[str, str]) -> None:
    entries = json.loads((headers / manifest).read_bytes())["entries"]
    v1 = V1((headers / next(e for e in entries if e["kind"] == "cozytensors")["blob"]["sha256"]).read_bytes())
    model = v1.configs.pop("model", None)
    sections = json.loads(model) if model else {}
    split = set(sections) == {c for c, _ in v1.components}
    for component, tensors in v1.components:
        name = names.get(component, component)
        folder = src if name == "model" else src / name
        doc, offset = {}, 0
        for key, dtype, shape in tensors:
            length = WIDTH[dtype]
            for n in shape:
                length *= n
            doc[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + length]}
            offset += length
        body = json.dumps(doc, separators=(",", ":")).encode()
        body += b" " * (-len(body) % 8)
        sparse(folder / "model.safetensors", struct.pack("<Q", len(body)) + body, offset)
        if component in v1.configs:
            (folder / "config.json").write_bytes(v1.configs[component])
        elif split:
            (folder / "config.json").write_text(json.dumps(sections[component]))
        elif name == "model" and model:
            (folder / "config.json").write_bytes(model)


def main() -> None:
    vectors, headers, out = (Path(a).resolve() for a in sys.argv[1:4])
    only = set(sys.argv[4:])
    out.mkdir(parents=True, exist_ok=True)
    for name in [*UPSTREAM, *LANES]:
        if only and name not in only:
            continue
        # The header records its sources as spelled; sparse files cost no disk.
        src = SOURCES / name
        shutil.rmtree(src, ignore_errors=True)
        sources: str | list[str] = str(src)
        if name in UPSTREAM:
            for origin, path in UPSTREAM[name].items():
                (src / path).parent.mkdir(parents=True, exist_ok=True)
                if origin.startswith("https://"):
                    (src / path).write_bytes(urllib.request.urlopen(origin).read())
                elif origin.endswith(".json"):
                    shutil.copy(vectors / origin, src / path)
                else:
                    sparse(src / path, *banked((vectors / origin).read_bytes()))
            # Loose files import one source each (SeedVR2's `.pt` constants); a folder as one.
            if all("/" not in path for path in UPSTREAM[name].values()):
                sources = sorted(str(p) for p in src.iterdir())
        else:
            from_lane(src, headers, *LANES[name])
        with tempfile.TemporaryDirectory(dir=out) as work:
            store = tensorfs.Store(work)
            ck = store.checkpoint(store.import_(sources))
            (out / f"{name}.cozytensors").write_bytes(ck.file("model.cozytensors"))
            print(name, {c: len(t) for c, t in ck.tensors.items()}, ck.metadata.get("converters"), flush=True)
        shutil.rmtree(src)


if __name__ == "__main__":
    main()
