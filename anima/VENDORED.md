# Vendored dependency closure

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. The wheel was built twice from the pinned merged Git commit at the platform-owned
`SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.22-py3-none-any.whl` | `cozy-runtime` `c1193159a5f729e6a63f20c634c139b732ffb3af` | `53e6fb37bba7f6cb0f6c1f68672880b4451dd711` | 757,436 | `47cbd0f3c55e3201648ecd12d4c82c3bb17299a305968d95f3350d1207f3316f` |

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
