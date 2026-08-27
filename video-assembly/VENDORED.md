# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.2-py3-none-any.whl` | `cozy-runtime` `430e052bc4f5bf5b41a65e0e71d19a7ffe7653e0` | 651,479 | `793d5c3f4c8fad4ede5a68a148134285b56dd21db4ba0b2f207a9caaa79b2f63` |

Two independent builds from that merged Runtime commit were byte-identical. The `media` extra
admits the exact locked PyAV/FFmpeg wheel; this endpoint owns no decoder, packet, path, or muxer.
