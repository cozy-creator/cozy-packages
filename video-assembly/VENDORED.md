# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.15-py3-none-any.whl` | `cozy-runtime` `0811500875b6850d21ffc66b106098fdd655a70c` | 733,494 | `3bf8fc0832553a90dd0c08207cc6f8c634ddf63724bb0fdebd4ae1a930d65352` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
