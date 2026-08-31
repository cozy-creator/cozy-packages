# Vendored first-party wheel

Anima consumes the same exact base-owned Cozy Runtime wheel as the other packages in this
repository. Tensorhub's canonical wheel materializer built it twice from the pinned Git commit at
the platform-owned `SOURCE_DATE_EPOCH=946684800`; the two outputs were byte-identical.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.15-py3-none-any.whl` | `cozy-runtime` `0811500875b6850d21ffc66b106098fdd655a70c` | 733,494 | `3bf8fc0832553a90dd0c08207cc6f8c634ddf63724bb0fdebd4ae1a930d65352` |

The wheel requires CPython `>=3.12,<3.13`; its optional `media` extra requires
`av>=18.1,<19`. Production placement resolves both dependencies against the exact base worker
image inventory and never installs a second Runtime copy into the package overlay.
