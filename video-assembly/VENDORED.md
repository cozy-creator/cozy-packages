# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.3-py3-none-any.whl` | `cozy-runtime` `a675e1085300d3b21e56f4787be630853c2e8393` | 710,824 | `1816d1b73ca8b0886afd0d511cb949f83dd64d2444b2c53645a77a650e742ef4` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
