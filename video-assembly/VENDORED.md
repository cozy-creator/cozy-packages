# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.29-py3-none-any.whl` | `cozy-runtime` `d0884f452859898f4aca44c43413884a40892284` | 745,253 | `d4f5c2afee924422cb6e974af9674be0ca63c7436113cde48b4aad73c1413b97` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
