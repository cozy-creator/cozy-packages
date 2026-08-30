# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.7-py3-none-any.whl` | `cozy-runtime` `a665b65c4a5c461c6e484d8b50fe736925af1d72` | 709,917 | `a73a61a85a6ce43da8f6288e14a12ce91672672a156b990a0ecf987e810bc37f` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
