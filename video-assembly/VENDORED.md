# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.2-py3-none-any.whl` | `cozy-runtime` `5617b4286ad26a75294a273b9a682fbe6d469f56` | 652,913 | `4df98d067f4c5699df78398b741381c0e96d731914ac4a92dbfa0cb88188100c` |

Two independent builds from that merged Runtime commit were byte-identical. The `media` extra
admits the exact locked PyAV/FFmpeg wheel; this endpoint owns no decoder, packet, path, or muxer.
