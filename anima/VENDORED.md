# Vendored first-party wheel

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. Tensorhub's canonical wheel materializer built it twice from the pinned Git commit at
the platform-owned `SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.9-py3-none-any.whl` | `cozy-runtime` `64c82a489c3a297aa4d779133480b0d7a7e0c5a6` | 693,563 | `511ead4219f41e876c9308a77d83e09b744522624fb3893db3c5a80a21a3cddb` |

The wheel requires CPython `>=3.14,<3.15`; its optional `media` extra requires
`av>=18.1,<19`. Production placement resolves both dependencies against the exact base worker
image inventory and never installs a second Runtime copy into the package overlay.
