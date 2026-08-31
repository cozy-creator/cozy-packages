# Vendored dependency closure

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. The wheel was built twice from the pinned merged Git commit at the platform-owned
`SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.22-py3-none-any.whl` | `cozy-runtime` `a54254ae6fd98798cc92dcf6ed8cf7df011434da` | `f3638e370ea87779f24941171c81ccb49ce3cf5b` | 752,030 | `926ffb7d7d6529b641ddde5c6b1cb13463d980b4f4d0084fa79fa55b2308f7ea` |

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
