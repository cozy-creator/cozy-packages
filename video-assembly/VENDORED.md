# Video assembly dependency closure

Cozy-owned libraries are exact local wheels; third-party packages are frozen by `uv.lock`.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.6-py3-none-any.whl` | `cozy-runtime` `793286c2f09c7207990fae581f1fb6d26f4abe50` | 703,000 | `d24214c8215b9d30c48a4f52f0acf564e8b426e8b16ce48fb96a9f9126b6af64` |

The `media` extra admits the exact locked PyAV/FFmpeg wheel; this package owns no decoder,
packet, path, or muxer.
