# Vendored first-party wheel

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. Tensorhub's canonical wheel materializer built it twice from the pinned Git commit at
the platform-owned `SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.10-py3-none-any.whl` | `cozy-runtime` `66c73a8738163e2e54105c10ee30f3b303b2d64e` | 698,262 | `94460296a7c6ad745d25693f8c67faa6543579ffb0779ee1f9b8056ef9a30ba8` |

The wheel requires CPython `>=3.14,<3.15`; its optional `media` extra requires
`av>=18.1,<19`. Production placement resolves both dependencies against the exact base worker
image inventory and never installs a second Runtime copy into the package overlay.
