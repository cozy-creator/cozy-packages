# serverless-endpoints (v2)

The product endpoint packages: MiniMax H3 (se-001/002/003), SDXL (se-008), the
`quality-judge` evaluation family (ev-003), and later waves. Each endpoint is an authored
package on `cozy_runtime.author` only (boundaries.md), installed via `cozy install`.
Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2)
`serverless-endpoints.md`; issues `tracker/serverless-endpoints/` (`se-*`).
No automated tests (decisions.md #160): verification = live runs + benchmarks.

## Layout

| path | what |
|---|---|
| `quality-judge/` | the eval judge family (ev-003): `judge`, `soft`, `pairwise`, `transcribe` |
| `quality-judge/endpoint.toml` | the in-repo DEFAULT bindings — never a source of truth |
| `quality-judge/endpoint.descriptor.json` | the committed surface contract; `describe --check` is the CI gate |
| `scripts/prepare.py` | HF checkpoint -> TensorFS store + artifact config (the artifact writer's stand-in) |
| `scripts/slice.py` | the local transport: ONE request through `run_slice`, no hub, no gRPC |
| `scripts/judge-live.py` | ev-003's live verification, on the RTX 4070 |
| `scripts/fence.py` | four structural fences |

## Running an endpoint locally

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python transformers pillow numpy msgspec soundfile /tmp/cozy-wheels/tensorfs-*.whl
uv pip install --python .venv/bin/python --no-deps <a pinned cozy-runtime checkout>
uv pip install --python .venv/bin/python -e ../cozy-eval          # the caller side only

nice -n 19 .venv/bin/python scripts/prepare.py                     # build the artifacts
nice -n 19 .venv/bin/cozy-runtime describe --dir quality-judge --check
nice -n 19 .venv/bin/python scripts/judge-live.py smoke
```

CI's environment holds cozy-runtime, msgspec and nothing heavy — no torch, no
transformers, no GPU — because `describe` reading the surface in that environment is the
PROOF that endpoint module scope stayed light. Reproduce it locally with a second venv so
the checks say the same thing here as they do in CI:

```bash
uv venv --python 3.13 .venv-check
uv pip install --python .venv-check/bin/python msgspec protobuf grpcio mypy ruff
uv pip install --python .venv-check/bin/python --no-deps <the pinned cozy-runtime checkout>

.venv-check/bin/python scripts/fence.py
.venv-check/bin/python -m mypy
.venv-check/bin/ruff check quality-judge/ scripts/
.venv-check/bin/cozy-runtime describe --dir quality-judge --check
```

Weights live OUTSIDE this repo (`~/cozy_v2/eval-models`) and are never committed. The
judge and transcriber checkpoints, their licences and their pinned revisions are declared
in `scripts/prepare.py` and bound in `endpoint.toml`; no endpoint module names any of them.

## The three rules an endpoint here obeys

**Code states capability, bindings state selection.** No module under an endpoint
directory names a repo, release, checkpoint digest or model revision. `fence.py` proves it.

**A request document is control, not data.** cozy-runtime's supervisor/executor seam caps
one frame at 64 KiB. Media arrives as typed ASSETS the kernel hydrates into the attempt
spool; an inline-bytes field large enough to be useful makes an endpoint unservable.

**Module scope stays light.** `describe` runs in a disposable container with no GPU and no
weights, so torch, transformers and PIL are imported inside the functions that need them.
