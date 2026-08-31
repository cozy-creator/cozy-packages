# SDXL dependency closure

`pyproject.toml` and `uv.lock` are this package's dependency authority. Runtime and Torch are
base-owned requirements: qualification matches their declared versions to the selected base worker
image and never installs a second copy in the package overlay. Runtime's optional
`cozy-runtime-cuda-kernels` wheel is also base-owned; SDXL activates only `media` and neither
resolves nor stores the kernel wheel.

## Cozy-owned wheel

The wheel was built twice from the pinned merged Git commit with
`SOURCE_DATE_EPOCH=946684800`; both outputs were byte-identical.

| file | source commit | source tree | bytes | SHA256 |
| --- | --- | --- | ---: | --- |
| `vendor/cozy_runtime-0.0.22-py3-none-any.whl` | `cozy-runtime` `a54254ae6fd98798cc92dcf6ed8cf7df011434da` | `f3638e370ea87779f24941171c81ccb49ce3cf5b` | 752,030 | `926ffb7d7d6529b641ddde5c6b1cb13463d980b4f4d0084fa79fa55b2308f7ea` |

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
