# Vendored dependency closure

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. The wheel was built twice from the pinned merged Git commit at the platform-owned
`SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.29-py3-none-any.whl` | `cozy-runtime` `d0884f452859898f4aca44c43413884a40892284` | `876ad7b0799a4d2e1ee32419af97563c611a599d` | 745,253 | `d4f5c2afee924422cb6e974af9674be0ca63c7436113cde48b4aad73c1413b97` |
| `vendor/tensorfs-0.0.7-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `tensorfs` `a9cfd601f3f8c0841907e2c2625021b463f0ef36` | `3c04d60b20f478d91801ca37278b5d23d1cd7237` | 2,360,087 | `a3296ef604061ce2984fdb74e329cf5e2daa69d7b45b924d5806b53ab9dc2521` |

The Runtime wheel requires CPython `>=3.12,<3.13`; its `media` extra requires `av>=18.1,<19` and
its `model-execution` extra owns exact TensorFS 0.0.7. The package declares the Runtime capability,
not a false direct TensorFS dependency. A full local venv installs both exact wheels. Production
placement resolves them against the exact base worker image inventory and prunes both base-owned
distributions from the package overlay.
Runtime's optional `cozy-runtime-cuda-kernels` wheel is likewise selected by the CUDA base image;
this package activates `media` and `model-execution` and neither resolves nor stores the kernel
wheel.

Anima also declares base-owned `torchvision>=0.28,<1`: Diffusers' Cosmos transformer guards its
`torchvision.transforms` import behind dependency availability, then requires it for the padding
mask on every Anima denoise step. The lock records the exact ABI-compatible member of the Torch
family while publication prunes it from the package overlay with the other base-owned wheels.

## Locked publication wheels

Creator's registry mirror admits only filenames ending exactly in `-py3-none-any.whl`. Anima's
pruned non-base closure also contains locked native `hf-xet`, PyYAML, regex, safetensors, and
tokenizers wheels plus Shellingham's compatible `py2.py3-none-any` wheel. Those six exact PyPI
objects are local direct dependencies under `vendor/`; `native-provenance.json` records their
source URLs, versions, lengths, and SHA-256 digests. `scripts/fence.py` joins those facts to
`pyproject.toml`, `uv.lock`, and the stored bytes so a future native transitive cannot fall back
into Creator's pure-wheel registry lane.
