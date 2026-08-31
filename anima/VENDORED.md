# Vendored dependency closure

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. Tensorhub's canonical wheel materializer built it twice from the pinned Git commit at
the platform-owned `SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.20-py3-none-any.whl` | `cozy-runtime` `ebdcb4a319dd1d6d2afa352d9dfe824c2bca2b57` | 751,483 | `ad0b814958ea738c454958eec4ebcefdff5b269979908bdf0a6ab1942ad6432a` |

The wheel requires CPython `>=3.12,<3.13`; its optional `media` extra requires
`av>=18.1,<19`. Production placement resolves both dependencies against the exact base worker
image inventory and never installs a second Runtime copy into the package overlay.

## Locked publication wheels

Creator's registry mirror admits only filenames ending exactly in `-py3-none-any.whl`. Anima's
pruned non-base closure also contains locked native `hf-xet`, PyYAML, regex, safetensors, and
tokenizers wheels plus Shellingham's compatible `py2.py3-none-any` wheel. Those six exact PyPI
objects are local direct dependencies under `vendor/`; `native-provenance.json` records their
source URLs, versions, lengths, and SHA-256 digests. `scripts/fence.py` joins those facts to
`pyproject.toml`, `uv.lock`, and the stored bytes so a future native transitive cannot fall back
into Creator's pure-wheel registry lane.
