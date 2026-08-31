# Vendored first-party wheel

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. Tensorhub's canonical wheel materializer built it twice from the pinned Git commit at
the platform-owned `SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.11-py3-none-any.whl` | `cozy-runtime` `7e71e1e560352af0c23b1118cc9c6bc7f2e0ddc6` | 705,603 | `9ad5642698d4879d4982152c5f88d0599a2b2c2f418f20c38bbb7d7fc5760334` |

The wheel requires CPython `>=3.12,<3.13`; its optional `media` extra requires
`av>=18.1,<19`. Production placement resolves both dependencies against the exact base worker
image inventory and never installs a second Runtime copy into the package overlay.
