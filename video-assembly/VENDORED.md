# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.3-py3-none-any.whl` | `cozy-runtime` `740247966cae4338b99dafac62761c42b817e018` | 679,122 | `5a18e2e9c84b4188943fca27b18275beed324486c7b0e81dd8fd4b10293c5d6d` |

Two independent builds from that merged Runtime commit were byte-identical. The `media` extra
admits the exact locked PyAV/FFmpeg wheel; this endpoint owns no decoder, packet, path, or muxer.
