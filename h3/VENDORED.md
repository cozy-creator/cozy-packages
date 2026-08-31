# H3 dependency and asset closure

This directory has one Python dependency authority: `pyproject.toml` plus
`uv.lock`. Cozy-owned libraries are immutable local wheels; third-party
packages are exact lock entries. TensorFS is the Runtime fill reader inside the isolated
package generation, and Pillow is not a direct package dependency (it remains a transitive
dependency of the pinned model stack).

## Cozy-owned wheels

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.8-py3-none-any.whl` | `cozy-runtime` `01cb96af87b9c71e09bcb2df46f708dcf18705ef` | 709,911 | `cfa8ceafb7a3026f674ba7bf8e7b0d0a6f7740f2e4ec54de71bd2c34f2339708` |
| `vendor/cozy_eval-2.3.0-py3-none-any.whl` | `cozy-eval` `fe8c7d2ead882dded2595e33af73101040792bd8` | 295,046 | `87830e9461f5b98b699e1ba7e1d2fec2dd37bea63d7b8ae8403506835e427a26` |
| `vendor/tensorfs-0.0.2-cp314-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `tensorfs` `1e6f5e6d1d9846836161939e0e687dea12ad8d4a` | 2,085,526 | `5d5420f9571778ae884daf6fe31bc649318014cd644d94b17a7c465929be5d93` |

Tensorhub's canonical first-party wheel materializer produced both artifacts from Git archives
of the exact commits in the table. It uses digest-pinned build containers, fixed in-container
paths, the platform epoch `946684800`, and normal CPython 3.14. Package repositories consume
those bytes; they do not maintain a competing wheel build command.

The source checkouts were at the commits in the table. The Runtime wheel
metadata declares `av>=18.1,<19` only for its `media` extra; this package asks
for `cozy-runtime[media]==0.0.8`.

## MiniMax H3 tokenizer and processor

The only accepted source is
`MiniMaxAI/MiniMax-H3@42ed227ee7df40d41602854ae760620d6eb651fe`.
The revision's own `model_index.json` binds `Qwen2TokenizerFast` and
`Qwen3VLProcessor`. The five files here are the minimal explicit-construction
closure: vocabulary, merges, tokenizer settings, and image/video processor
settings. They are regular copies of that exact immutable snapshot, not cache
symlinks.

These package assets are the one unweighted authority. The bound artifact
owns weighted component configs and must not repeat the tokenizer vocabulary
or processor settings. Runtime `Config` intentionally validates mapping keys;
a raw tokenizer vocabulary contains path-like tokens and is not a legal
component-config mapping. The package does not bypass that validation.

The weighted text-encoder config carries the closed `cozy_h3` extension. Its member and
exact field set supply the type. It fixes the source architecture to Qwen3-VL, retains
decoder layers 0–49, selects pre-norm hidden state 50, and declares
the language-model head absent. The package keeps the ordinary Transformers `.model`
and `.config` surface while final norm/head become parameterless identities. Runtime's
construction census therefore contains exactly 902 BF16 destinations; the removed tail
is not a hidden fetch or execution mode.

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
| `tokenizer/tokenizer_config.json` | `a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68` |
| `tokenizer/vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `processor/preprocessor_config.json` | `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516` |
| `processor/video_preprocessor_config.json` | `7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13` |

With network access disabled and CUDA hidden, Transformers 5.16.1 explicitly
constructed `Qwen2Tokenizer` (vocabulary size 151643) and `Qwen3VLProcessor`
from only these files. Token IDs, chat template, and processor configuration
matched the full snapshot construction.

## Third-party binary closure

The package declares this compatible model stack:

- `diffusers==0.40.0`
- `transformers==5.16.1`
- `torch>=2.13,<3`
- `torchvision==0.28.0`

The lock freezes that compatible range to Torch 2.13.0 and its CUDA 13 peers. `uv lock`
resolves the Linux x86-64 closure. A new Python 3.14 environment installed
all 65 non-project packages from the populated uv cache with this command and
no network access:

```sh
cd h3
uv lock --check

offline_env=$(mktemp -d /tmp/h3-lock-offline.XXXXXX)
rmdir "$offline_env"
UV_PROJECT_ENVIRONMENT="$offline_env" \
  uv sync --locked --offline --no-install-project
```

That proves the complete locked environment can be replayed offline from the
build cache; a fresh builder must still populate its third-party wheel cache or
mirror before entering offline mode.

The Runtime media extra resolves to this exact PyAV wheel:

| package | wheel | SHA256 |
| --- | --- | --- |
| `av==18.1.0` | `av-18.1.0-cp311-abi3-manylinux_2_28_x86_64.whl` | `8a032e8d8ebc73dec079364b9b4a6837638a2d106e8472314e685ffbf163e700` |

An isolated Python 3.14 environment installed the Runtime media closure from a
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

The output path is covered separately: encoder construction succeeded for
`libx264` and `aac`, and the vendored Runtime's real `Outputs.save_video`
wrote a 2,208-byte MP4 containing H264 video and AAC audio. This is a tiny
encode/mux mechanism receipt, not an H3 generation or quality proof.
