# Vendored dependency closure

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. The wheel was built twice from the pinned merged Git commit at the platform-owned
`SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.26-py3-none-any.whl` | `cozy-runtime` `b6c65d75dc1fbb09af27e137b430f26bb21cedac` | `a72ceccc4d09f50a62e68487181373812b5ceb21` | 749,670 | `9422821362cdba58c9b7dc1d9aeb881d140f28ab4caedba0cbed6c39c6fca7c0` |
| `vendor/tensorfs-0.0.6-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `tensorfs` `ba40d92bc68e33f10c92848733d407ab887d11c7` | `00e762f61c2e0bc01339b53b0d3ee17d8cae2567` | 2,347,973 | `83bbf80e3279d1f5ced912731a2a259ad557c483b8cb0123fc4795c2383d8b1f` |

The Runtime wheel requires CPython `>=3.12,<3.13`; its `media` extra requires `av>=18.1,<19` and
its `model-execution` extra owns exact TensorFS 0.0.6. The package declares the Runtime capability,
not a false direct TensorFS dependency. A full local venv installs both exact wheels. Production
placement resolves them against the exact base worker image inventory and prunes both base-owned
distributions from the package overlay.
Runtime's optional `cozy-runtime-cuda-kernels` wheel is likewise selected by the CUDA base image;
this package activates `media` and `model-execution` and neither resolves nor stores the kernel
wheel.

## Locked publication wheels

Creator's registry mirror admits only filenames ending exactly in `-py3-none-any.whl`. Anima's
pruned non-base closure also contains locked native `hf-xet`, PyYAML, regex, safetensors, and
tokenizers wheels plus Shellingham's compatible `py2.py3-none-any` wheel. Those six exact PyPI
objects are local direct dependencies under `vendor/`; `native-provenance.json` records their
source URLs, versions, lengths, and SHA-256 digests. `scripts/fence.py` joins those facts to
`pyproject.toml`, `uv.lock`, and the stored bytes so a future native transitive cannot fall back
into Creator's pure-wheel registry lane.
