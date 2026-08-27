# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.2-py3-none-any.whl` | `cozy-runtime` `e4725826df490e5446e99ed9407547c35f85d7df` | 654,578 | `edb1d4b727370f9aa9b557e4048cee21ee92d3416c52198cbddf70d2fe22e587` |

Two independent builds from that merged Runtime commit were byte-identical. The `media` extra
admits the exact locked PyAV/FFmpeg wheel; this endpoint owns no decoder, packet, path, or muxer.
