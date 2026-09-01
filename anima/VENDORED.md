# Vendored dependency closure

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. The wheel was built twice from the pinned merged Git commit at the platform-owned
`SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.23-py3-none-any.whl` | `cozy-runtime` `b286b18404a56d87ef89f141019e7bb7e4f3f9c1` | `11a9478141aeb034d8caa5b8a59782a518377223` | 750,585 | `eabb44ebb99d30627b41d37549f3774f770834505d37c46cd03fc53f63d7826d` |

The wheel requires CPython `>=3.12,<3.13`; its optional `media` extra requires
`av>=18.1,<19`. Production placement resolves both dependencies against the exact base worker
image inventory and never installs a second Runtime copy into the package overlay.
Runtime's optional `cozy-runtime-cuda-kernels` wheel is likewise selected by the CUDA base image;
this package activates only `media` and neither resolves nor stores the kernel wheel.

## Locked publication wheels

Creator's registry mirror admits only filenames ending exactly in `-py3-none-any.whl`. Anima's
pruned non-base closure also contains locked native `hf-xet`, PyYAML, regex, safetensors, and
tokenizers wheels plus Shellingham's compatible `py2.py3-none-any` wheel. Those six exact PyPI
objects are local direct dependencies under `vendor/`; `native-provenance.json` records their
source URLs, versions, lengths, and SHA-256 digests. `scripts/fence.py` joins those facts to
`pyproject.toml`, `uv.lock`, and the stored bytes so a future native transitive cannot fall back
into Creator's pure-wheel registry lane.
