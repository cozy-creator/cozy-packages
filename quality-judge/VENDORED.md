# Quality judge dependency closure

This directory has one Python dependency authority: `pyproject.toml` plus `uv.lock`.
Cozy-owned libraries are immutable local wheels; third-party packages are exact lock
entries. Before se-019 this package had neither file, so it could not produce a
`ResolvedWheelSet` and could not publish at all.

## Cozy-owned wheels

Byte-identical copies of the wheels H3 vendors — the same Runtime build serves both
packages, so the CAS stores one copy.

| file | source commit | bytes | SHA256 |
| --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.29-py3-none-any.whl` | `cozy-runtime` `d0884f452859898f4aca44c43413884a40892284` | 745,253 | `d4f5c2afee924422cb6e974af9674be0ca63c7436113cde48b4aad73c1413b97` |
| `vendor/tensorfs-0.0.7-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `tensorfs` `a9cfd601f3f8c0841907e2c2625021b463f0ef36` | 2,360,087 | `a3296ef604061ce2984fdb74e329cf5e2daa69d7b45b924d5806b53ab9dc2521` |

TensorFS is the Runtime fill reader inside the isolated package generation. Cozy-eval is
NOT here: `quality_judge.py` imports nothing from it — `scripts/judge-live.py` is a driver
that runs cozy-eval's prompt builders and parsers on the CALLER's side, outside this
environment.

## Locked publication wheels

Creator's registry mirror admits only filenames ending exactly in `-py3-none-any.whl`. The
judge's pruned non-base closure also contains locked native `hf-xet`, PyYAML, regex, safetensors,
and tokenizers wheels plus Shellingham's compatible `py2.py3-none-any` wheel. Those six exact
PyPI objects are local direct dependencies under `vendor/`; `native-provenance.json` records their
source URLs, versions, lengths, and SHA-256 digests. `scripts/fence.py` joins those facts to
`pyproject.toml`, `uv.lock`, and the stored bytes so a future native transitive cannot fall back
into Creator's pure-wheel registry lane.

## Third-party binary closure

The package declares this compatible model stack:

- `transformers==5.16.1`
- `torch>=2.13,<3`

The lock freezes that range to Torch 2.13.0 and its CUDA 13 peers. H3 and the judge resolve
the same exact family. Torch's compiled extensions link against one C++ ABI, so the
platform supplies one internally consistent family even though package placement accepts
compatible Torch releases in the declared range.

Neither torchvision nor torchaudio is in this closure. The judge's image path receives
frames already decoded by the Runtime media extra and hands Pillow images to the
Transformers processor; the transcriber's audio path is resampled in numpy against
Whisper's one input rate. Pillow, numpy and tokenizers are DIRECT dependencies rather than
inherited ones — `quality_judge.py` imports all three by name.

`uv lock` resolves the Linux x86-64 closure on Python 3.12. A new environment installed
all 63 non-project packages from the populated uv cache with no network access:

```sh
cd quality-judge
uv lock --check

offline_env=$(mktemp -d /tmp/quality-judge-lock-offline.XXXXXX)
rmdir "$offline_env"
UV_PROJECT_ENVIRONMENT="$offline_env" \
  uv sync --locked --offline --no-install-project
```

That environment then imported the real stack and the package module:
`torch 2.13.0`, `transformers 5.16.1`, and `quality_judge:app` with
its four entrypoints (`judge`, `soft`, `pairwise`, `transcribe`). A fresh builder must
still populate its third-party wheel cache or mirror before entering offline mode.

The Runtime media extra resolves to the same PyAV wheel H3 locks, `av==18.1.0`.

## Publication

The package-domain hardcut changes the project wheel and every document above it. The retired
pre-hardcut receipt is intentionally not carried forward; Tensorhub must publish and qualify a
fresh package release from this exact lock and source tree.
