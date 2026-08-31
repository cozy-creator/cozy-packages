# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.11-py3-none-any.whl` | `cozy-runtime` `7e71e1e560352af0c23b1118cc9c6bc7f2e0ddc6` | 705,603 | `9ad5642698d4879d4982152c5f88d0599a2b2c2f418f20c38bbb7d7fc5760334` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
