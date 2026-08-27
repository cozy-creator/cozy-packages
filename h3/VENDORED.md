# H3 dependency and asset closure

This directory has one Python dependency authority: `pyproject.toml` plus
`uv.lock`. Cozy-owned libraries are immutable local wheels; third-party
packages are exact lock entries. TensorFS is not an endpoint dependency, and
Pillow is not a direct endpoint dependency (it remains a transitive dependency
of the pinned model stack).

## Cozy-owned wheels

| file | source commit | SHA256 |
| --- | --- | --- |
| `vendor/cozy_runtime-0.0.1-py3-none-any.whl` | `cozy-runtime` `1bbf0bd934fe605db70401566ac4c3f12dfbe6c1` | `3af60632326a34f336571f4cc46c0e0288a86c96f61aa0443208c1b342a4dca8` |
| `vendor/cozy_eval-2.3.0-py3-none-any.whl` | `cozy-eval` `61560ff5cc0bce2403b7e2bc9e6bff753be58ca1` | `8b2af65159e9b5446d7fb1025988ba1aea74f14aea0bb27668c73260344f23a8` |

Both checked-in wheels reproduced byte-for-byte on 2026-08-26 with uv 0.9.18:

```sh
runtime_out=$(mktemp -d /tmp/cozy-runtime-wheel.XXXXXX)
eval_out=$(mktemp -d /tmp/cozy-eval-wheel.XXXXXX)
uv build --wheel --out-dir "$runtime_out" /home/fidika/cozy_v2/cozy-runtime
uv build --wheel --out-dir "$eval_out" /home/fidika/cozy_v2/cozy-eval
sha256sum "$runtime_out"/*.whl "$eval_out"/*.whl
```

The source checkouts were at the commits in the table. The Runtime wheel
metadata declares `av>=18.1,<19` only for its `media` extra; this endpoint asks
for `cozy-runtime[media]==0.0.1`.

## MiniMax H3 tokenizer and processor

The only accepted source is
`MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.
The revision's own `model_index.json` binds `Qwen2TokenizerFast` to
`tokenizer/` and `Qwen3VLProcessor` to `processor/`. Files here are regular
copies of that exact immutable Hugging Face snapshot, not cache symlinks.

The initially supplied name `MiniMaxAI/MiniMax-Hailuo-2.3` was rejected: both
its model API and exact-revision tree returned HTTP 401 without credentials,
while the existing exact-revision official snapshot and its model metadata use
`MiniMaxAI/MiniMax-H3`. The rejected name is not retained as an alias or
fallback. A live re-fetch of the accepted gated repository still requires an
authorized Hugging Face token; verification below used the already-fetched
exact-revision snapshot.

| file | SHA256 |
| --- | --- |
| `tokenizer/merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `tokenizer/tokenizer.json` | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `tokenizer/tokenizer_config.json` | `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68` |
| `tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `processor/chat_template.json` | `5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce` |
| `processor/merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `processor/preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `processor/tokenizer.json` | `a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7` |
| `processor/tokenizer_config.json` | `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68` |
| `processor/video_preprocessor_config.json` | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |
| `processor/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

With network access disabled and CUDA hidden, Transformers 5.16.1 loaded the
local files as `Qwen2Tokenizer` (vocabulary size 151643) and
`Qwen3VLProcessor` with the same tokenizer vocabulary.

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
