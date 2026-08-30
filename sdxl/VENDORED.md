# SDXL dependency closure

`pyproject.toml` and `uv.lock` are this package's dependency authority. Runtime and Torch are
base-owned requirements: qualification matches their declared versions to the selected base worker
image and never installs a second copy in the package overlay.

## Cozy-owned wheel

| file | bytes | SHA256 |
| --- | ---: | --- |
| `vendor/cozy_runtime-0.0.6-py3-none-any.whl` | 703,000 | `d24214c8215b9d30c48a4f52f0acf564e8b426e8b16ce48fb96a9f9126b6af64` |

## Generation stack

- `torch>=2.13,<3` (the selected base supplies one exact compatible build)
- `diffusers==0.40.0`
- `transformers==5.16.1`
- `hidiffusion==0.1.10`

Full HiDiffusion is the default for the square 1024x1024 bucket. Native non-square buckets use
ordinary SDXL after live 1344x768 HiDiffusion duplicated a singular subject and window-attention-only
produced a duplicated head at 1024x1024; future high-resolution lanes remain a separate experiment.
The package owns the denoising loop,
so it sets the UNet's total timestep count and resets every patched module's request-local timestep
state before step zero. A later request cannot inherit a partial or canceled request's counters.

`uv lock` resolves the Linux x86-64 closure for Python 3.14. Verify the lock and static package
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
