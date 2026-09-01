# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.37-py3-none-any.whl` | `cozy-runtime` `9969b73d4521997508a6e3894f1e3199d3765598` | 753,255 | `554051728d6ea831e138e1c8b8f185f7b4081ec02abed1e6d0503f13cfc72e2e` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
