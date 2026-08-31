# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.10-py3-none-any.whl` | `cozy-runtime` `66c73a8738163e2e54105c10ee30f3b303b2d64e` | 698,262 | `94460296a7c6ad745d25693f8c67faa6543579ffb0779ee1f9b8056ef9a30ba8` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
