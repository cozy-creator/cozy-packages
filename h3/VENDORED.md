# H3 dependency and asset closure

This directory has one Python dependency authority: `pyproject.toml` plus
`uv.lock`. Cozy-owned libraries are immutable local wheels; third-party
packages are exact lock entries. TensorFS is not an endpoint dependency, and
Pillow is not a direct endpoint dependency (it remains a transitive dependency
of the pinned model stack).

## Cozy-owned wheels

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.1-py3-none-any.whl` | `cozy-runtime` `1bbf0bd934fe605db70401566ac4c3f12dfbe6c1` | 583,723 | `3af60632326a34f336571f4cc46c0e0288a86c96f61aa0443208c1b342a4dca8` |
| `vendor/cozy_eval-2.3.0-py3-none-any.whl` | `cozy-eval` `61560ff5cc0bce2403b7e2bc9e6bff753be58ca1` | 293,042 | `8b2af65159e9b5446d7fb1025988ba1aea74f14aea0bb27668c73260344f23a8` |

Both checked-in wheels reproduced byte-for-byte on 2026-08-26 with uv 0.9.18:

```sh
runtime_src=/absolute/path/to/cozy-runtime
eval_src=/absolute/path/to/cozy-eval
runtime_out=$(mktemp -d /tmp/cozy-runtime-wheel.XXXXXX)
eval_out=$(mktemp -d /tmp/cozy-eval-wheel.XXXXXX)
uv build --wheel --out-dir "$runtime_out" "$runtime_src"
uv build --wheel --out-dir "$eval_out" "$eval_src"
sha256sum "$runtime_out"/*.whl "$eval_out"/*.whl
```

The source checkouts were at the commits in the table. The Runtime wheel
metadata declares `av>=18.1,<19` only for its `media` extra; this endpoint asks
for `cozy-runtime[media]==0.0.1`.

## Model configuration authority

Tokenizer vocabulary, merges, tokenizer settings, image/video processor settings, and chat
template belong to the exact bound model artifact's `Config`. The endpoint package carries no
second copy and performs no model-hub lookup. A missing or malformed mapping refuses model
construction.

The final job-001 artifact must record the stored-byte identities and provenance of those mappings
against the accepted official `MiniMaxAI/MiniMax-H3` revision
`42ed227ee7df40d41602854ae760620d6eb651fe`. The initially supplied
`MiniMaxAI/MiniMax-Hailuo-2.3` name returned HTTP 401 and contradicted the cached official snapshot;
it is not an alias or fallback.

## Third-party binary closure

The model stack is pinned to:

- `diffusers==0.40.0`
- `transformers==5.16.1`
- `torch==2.9.1+cu129`
- `torchvision==0.24.1+cu129`
- `torchaudio==2.9.1+cu129`

Torchvision uses its compatible 0.24.1 release number; there is no
`torchvision==2.9.1+cu129` artifact in the selected CUDA index. `uv lock`
resolves 66 packages for Linux x86-64. A new Python 3.12 environment installed
all 65 non-project packages from the populated uv cache with this command and
no network access:

```sh
cd h3
uv lock
uv lock --check

offline_env=$(mktemp -d /tmp/h3-lock-offline.XXXXXX)
rmdir "$offline_env"
UV_PROJECT_ENVIRONMENT="$offline_env" \
  uv sync --frozen --offline --no-install-project
```

That proves the complete locked environment can be replayed offline from the
build cache; a fresh builder must still populate its third-party wheel cache or
mirror before entering offline mode.

The Runtime media extra resolves to this exact PyAV wheel:

| package | wheel | SHA256 |
| --- | --- | --- |
| `av==18.1.0` | `av-18.1.0-cp311-abi3-manylinux_2_28_x86_64.whl` | `8a032e8d8ebc73dec079364b9b4a6837638a2d106e8472314e685ffbf163e700` |

An isolated Python 3.12 environment installed the Runtime media closure from a
local wheelhouse using `--offline --no-index`. Importing the installed Runtime
and PyAV succeeded. The wheel reports these embedded library versions:

```text
libavcodec 62.28.102
libavdevice 62.3.102
libavfilter 11.14.102
libavformat 62.12.102
libavutil 60.26.102
libswresample 6.3.102
libswscale 9.5.102
```

Decoder construction succeeded for PNG, MJPEG/JPEG, GIF, BMP, TIFF, WebP,
AV1/AVIF, H.264, HEVC, VP8, VP9, MPEG-4, ProRes, AAC, Opus, Vorbis, FLAC, MP3,
PCM S16LE, and PCM F32LE. Input demuxer construction succeeded for MOV/MP4,
Matroska/WebM, WAV, Ogg, FLAC, MP3, and image sequences. This is the PyAV
wheel's bundled FFmpeg capability census; it does not rely on a system
`ffmpeg` executable.
