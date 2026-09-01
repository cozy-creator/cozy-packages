# SDXL dependency closure

`pyproject.toml` and `uv.lock` are this package's dependency authority. Runtime, TensorFS, and Torch are
base-owned requirements: qualification matches their declared versions to the selected base worker
image and never installs a second copy in the package overlay. Runtime's optional
`cozy-runtime-cuda-kernels` wheel is also base-owned. SDXL activates Runtime's `media` and
`model-execution` capabilities; Runtime owns the transitive TensorFS requirement, and package code
declares no false direct TensorFS dependency.

## Cozy-owned wheel

The wheel was built twice from the pinned merged Git commit with
`SOURCE_DATE_EPOCH=946684800`; both outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.26-py3-none-any.whl` | `cozy-runtime` `b6c65d75dc1fbb09af27e137b430f26bb21cedac` | `a72ceccc4d09f50a62e68487181373812b5ceb21` | 749,670 | `9422821362cdba58c9b7dc1d9aeb881d140f28ab4caedba0cbed6c39c6fca7c0` |
| `vendor/tensorfs-0.0.6-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `tensorfs` `ba40d92bc68e33f10c92848733d407ab887d11c7` | `00e762f61c2e0bc01339b53b0d3ee17d8cae2567` | 2,347,973 | `83bbf80e3279d1f5ced912731a2a259ad557c483b8cb0123fc4795c2383d8b1f` |

## Locked publication wheels

Creator's registry mirror admits only filenames ending exactly in `-py3-none-any.whl`. SDXL's
pruned non-base closure also contains locked native `hf-xet`, PyYAML, regex, safetensors, and
tokenizers wheels plus Shellingham's compatible `py2.py3-none-any` wheel. Those six exact PyPI
objects are local direct dependencies under `vendor/`; `native-provenance.json` records their
source URLs, versions, lengths, and SHA-256 digests. `scripts/fence.py` joins those facts to
`pyproject.toml`, `uv.lock`, and the stored bytes so a future native transitive cannot fall back
into Creator's pure-wheel registry lane.

## Generation stack

- `torch>=2.13,<3` (the selected base supplies one exact compatible build)
- `diffusers==0.40.0`
- `transformers==5.16.1`
- `hidiffusion==0.1.10`

The default development binding selects `paul/wai-illustrious@17.0.0`, lane `bf16`,
and requests default to CFG 7 and `hidiffusion: true`.

Package release `2.0.0` intentionally marks the pre-launch result-schema break that added
`hidiffusion_applied`; a published 1.x result schema cannot gain a field under the same major.

Full HiDiffusion is applied only when the request enables it and the effective output geometry is
square. The result reports that decision as `hidiffusion_applied`. Native non-square buckets use
ordinary SDXL after live 1344x768 HiDiffusion duplicated a singular subject and
window-attention-only produced a duplicated head at 1024x1024; future high-resolution lanes remain
a separate experiment.
The package owns the denoising loop,
so it sets the UNet's total timestep count and resets every patched module's request-local timestep
state before step zero. A later request cannot inherit a partial or canceled request's counters.

`uv lock` resolves the Linux x86-64 closure for Python 3.12. Verify the lock and static package
surface with:

```sh
cd sdxl
uv lock --check
uv run ruff check sdxl
uv run mypy --config-file ../pyproject.toml sdxl
uv run cozy-runtime --json describe
```

The package must be republished and requalified after this change: the project wheel, dependency
closure, factory identity, and model-construction contract all changed.
