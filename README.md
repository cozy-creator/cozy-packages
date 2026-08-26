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
| `h3/h3_arch/` | the current H3 release's one executable graph: the native community-curve port (see `NOTICE`) |
| `sdxl/` | the SDXL launch endpoint (se-008): `generate`, text to image, four components |
| `sdxl/tokenizer`, `sdxl/tokenizer_2` | the two CLIP vocabularies this endpoint BUNDLES — its own asset, like the model library it imports |
| `quality-judge/` | the eval judge family (ev-003): `judge`, `soft`, `pairwise`, `transcribe` |
| `*/endpoint.toml` | the in-repo DEFAULT bindings — never a source of truth |
| `*/endpoint.descriptor.json` | the committed surface contract; `describe --check` is the CI gate |
| `scripts/prepare.py` | HF checkpoint -> TensorFS store + artifact config (the artifact writer's stand-in) |
| `scripts/slice.py` | the local transport: ONE request through `run_slice`, no hub, no gRPC |
| `scripts/judge-live.py` | ev-003's live verification, on the RTX 4070 |
| `scripts/sdxl-release.sh` | build the SDXL release archive `cozy install --from --digest` verifies |
| `scripts/pack.py` | tree -> release archive (the pre-hub stand-in for `cozy deploy`) |
| `scripts/sdxl-live.py` | se-008's live verification, on the RTX 4070 |
| `scripts/h3-keys.py` | the port's graph against the community carrier's headers — key-exact, $0, no GPU |
| `scripts/h3-oracle.py` | the pinned ComfyUI differential oracle; never imported by an endpoint, never in a release |
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
.venv-check/bin/ruff check quality-judge/ sdxl/ scripts/
.venv-check/bin/cozy-runtime describe --dir quality-judge --check
.venv-check/bin/cozy-runtime describe --dir sdxl --check
```

Weights live OUTSIDE this repo (`~/cozy_v2/eval-models` for the judge, `/tmp/cozy-sdxl4`
for SDXL) and are never committed. Checkpoints, their licences and their pinned revisions
are declared in `scripts/prepare.py` and bound in each `endpoint.toml`; no endpoint module
names any of them.

## Serving SDXL through the product

The launch endpoint is installed and run the way a user's endpoint is — as a release
archive through `cozy install`, not from this tree:

```bash
nice -n 19 ./scripts/sdxl-release.sh                              # the bf16/fp16 rung
nice -n 19 ./scripts/sdxl-release.sh --endpoint cozy/sdxl-fp8 --artifact-release fp8

nice -n 19 .venv-check/bin/python scripts/sdxl-live.py product arms variants clamp
nice -n 19 .venv/bin/python       scripts/sdxl-live.py judge     # the caller-side venv
```

The release pins its peers by read-only `git archive` — cozy-runtime and the compiled
`tensorfs` facade, whose ENCODING REGISTRY is what decides which lanes the endpoint can
read. It declares `torch`/`diffusers`/`transformers` as its OWN dependencies: a model
architecture is the endpoint's, never the runtime's.

## H3: where the architecture comes from, and who checks it

Decision #593 fixed the release boundary after the official candidate was implemented but
found not executable end to end. This release ships only the graph its bound artifact fills
and its action can run. Evidence sources remain ordered:

1. the **official model release** — its configs, its docs and its licence;
2. the **pinned diffusers/transformers implementation** — an upstream statement and oracle,
   not an artifact-selected second serving graph;
3. **DiffSynth-Studio** — an independent second opinion, never a dependency;
4. **ComfyUI** — a foreign-format producer and a black-box speed baseline, and nothing
   else. Never serving architecture, never pipeline semantics, never product vocabulary.

### The two differential oracles

Same model, same weights, two independent implementations to disagree with. A seam is only
as proven as the number of unrelated codebases that reproduce it.

| oracle | what it is | how it is used |
|---|---|---|
| ComfyUI v0.33.0 `comfy/ldm/minimax/` | `scripts/h3-oracle.py`, pinned | a checkout on a rented card. Never imported by an endpoint, never in a release archive |
| DiffSynth-Studio `diffsynth/pipelines/minimax_h3_audio_video.py` (modelscope) | RECORDED, not vendored | an independent third reading of the same seams |

**The comparison protocol**, in the order a disagreement is cheapest to localise:

1. **On this box, $0** — construct on `meta` and diff the census against banked headers;
   check the temporal geometry, the sigma grid, the evaluation count and the velocity sign
   against upstream's own arithmetic. `scripts/h3-keys.py`, `h3-conform.py`, and
   `h3-vision-conform.py` are the current zero-dollar proof set.
2. **Component-level, on a card** — same weights, same input, one component at a time:
   VAE encode/decode round trips, one DiT block, the rope table, the modulation. se-001 ran
   this against ComfyUI for $0.21 and it found a real defect.
3. **Whole-seam, on a card** — same prompt and seed, every intermediate compared in order:
   token ids, text-encoder states, packed rows, raw heads, latent-shaped velocity, first
   updated latent, final latent, decoded video and audio. **This is the step se-002 skipped**,
   and the four defects #522 names all lived BETWEEN components where component-level
   agreement could not see them.
4. **Output**, last and never first — full-length, gate-passing, and viewed.

A number produced by only one of the three implementations is a measurement, not a proof.

### Licensing

`LICENSE` is MIT. `NOTICE` records what in this tree is not: diffusers and transformers are
Apache-2.0 dependencies and are not vendored, and `h3/h3_arch/` is adapted from GPL-3.0
ComfyUI source, which MIT does not cover. Decision #593 deleted the incomplete in-wheel
replacement; a future official transition must land atomically with its exact artifact and
complete output proof, then delete this port. The model's own licence is separate from all
of them, carries a territory restriction, and is an open owner ruling — read `NOTICE`
before serving H3 anywhere.

## The three rules an endpoint here obeys

**Code states capability, bindings state selection.** No module under an endpoint
directory names a repo, release, checkpoint digest or model revision. `fence.py` proves it.

**A request document is control, not data.** cozy-runtime's supervisor/executor seam caps
one frame at 64 KiB. Media arrives as typed ASSETS the kernel hydrates into the attempt
spool; an inline-bytes field large enough to be useful makes an endpoint unservable.

**Module scope stays light.** `describe` runs in a disposable container with no GPU and no
weights, so torch, transformers and PIL are imported inside the functions that need them.
