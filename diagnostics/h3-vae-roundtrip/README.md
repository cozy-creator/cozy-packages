# Optional H3 diagnostics

This is the unpublished 1.0.10 candidate in PR114. It depends on the normal
`minimax-h3==1.1.11` wheel and public Runtime >=0.2.34. Ordinary
`paul/minimax-h3` inference is unchanged and has no new quality gate.

| Entrypoint | Computation | Additional evidence |
|---|---|---|
| `reference_activations` | Original reference preprocessing and first denoise step | Small activation JSON; no video or final latents |
| `reference_trace` | Complete original reference-video inference | Exact returned video result, activation JSON, final latents in safetensors |
| `roundtrip` | Existing 22/345-frame VAE reconstruction | Source/reconstruction images, video, resident tensor hashes and grid observation |

Both reference diagnostics accept the normal H3 reference request and require an
explicit seed. Select the reviewed exact checkpoint through the normal
`model.model` override; there is deliberately no default checkpoint for them.
Use identical prompt, seed, reference digests/order and reference-image short edge
for BF16/FP8 pairs. An unseeded diagnostic is refused during preflight.

## One inference implementation and one component scope

Both actions call the imported `h3.reference_media_to_video` function. The model
inherits H3's original sampling methods and their ordinary Runtime component
scopes. A thin pipeline subclass installs observers around `super().denoise` and
wraps its existing step callback. It supplies no sampler, changes no argument or
tensor, and runs the inherited pipeline constructor. Hooks and the request-local
ContextVar are removed on success, cancellation or exception.

The first-step action raises its own sentinel only from the ordinary callback
after the first solver update. Its outer diagnostic handler consumes that sentinel
and returns `completed_steps=1`. It skips VAE decoding and makes no claim about a
final video or final latents. Other exceptions still propagate. The complete action
allows every original step and returns H3's exact video result under `inference`.
Capture adds synchronization overhead; use ordinary inference for timing benchmarks.

The old gate-first override and custom short sampler are removed from this new
revision. The already-published 1.0.9 remains unchanged as historical evidence.

## Activation document: h3.activation-samples/1

The JSON contains `mode`, `component`, `selected_steps`, `selected_blocks`,
`completed_steps`, `final_latents_present`, `samples`, `final_latents` geometry
and `provenance`. Provenance records the request ID, Runtime-owned checkpoint ref,
package/runtime/torch/diffusers versions, prompt, seed, reference kinds/digests/sizes
and reference-image short edge. It contains no asset capability or local path.

For the 50-block H3 model at the request's `steps` (default 30), complete mode observes
blocks 0/24/49 at steps 0/15/29; first-step mode observes the same blocks at step 0. Block
positions are computed from the actual block count. Each forward selects up to
eight evenly spaced packed rows **per modality**, sixteen evenly spaced channels,
and all batches (at most two). Indices include both ends when there is more than
one coordinate. Short modalities include every row. Selection uses integer
arithmetic and consumes no RNG.

Each sample names its zero-based step, forward ordinal within that step, module,
original shape/dtype/device, batch/channel indices, packed row indices, modality
labels, rotary positions, timestep indices/tags and actual timestep values.
`values[batch][row][channel]` contains the selected scalars represented as float32.
A comparator must require matching coordinates and request conditions before
computing sampled MAE/RMSE/cosine or another fidelity measure. These samples do
not measure every activation or establish general output quality.

At most four forwards per captured step are allowed. Rows/channels are gathered
on the device before the small CPU copy; the full activation sequence is never
copied. JSON is limited to 1 MiB. Complete-run video/audio latents are copied once
in their original dtype, capped at 96 MiB before copying, and serialized with the
standard safetensors library into a FileAsset capped at 128 MiB. Partial runs
cannot label any tensor as final. All artifacts use Runtime Outputs.

This package produces evidence, not a Tensorhub assessment or a pass/fail quality
verdict. Tier-2 comparisons use matching activation samples; Tier-3 comparisons
and aesthetic/prompt-adherence checks use the separately generated complete videos
through cozy-eval. A first-step probe cannot substitute for an output comparison.

## Build and bounded CPU proof

```sh
uv build --project minimax-h3 --wheel --out-dir diagnostics/h3-vae-roundtrip/dependencies
uv lock --check --project diagnostics/h3-vae-roundtrip
uv sync --locked --project diagnostics/h3-vae-roundtrip
diagnostics/h3-vae-roundtrip/.venv/bin/python scripts/h3-activation-proof.py
diagnostics/h3-vae-roundtrip/.venv/bin/cozy-runtime --json --dir diagnostics/h3-vae-roundtrip describe
```

The dependency's twelve payload files were checked against the installed public
1.1.11 package. The CPU proof checks exact sampled coordinates, bitwise unchanged
outputs/RNG, repeatability, exception cleanup, first-step stopping through the real
H3 Model/Runtime wrapper, rejection of partial final-latent claims, and safetensors
round-trip fidelity. It uses tiny synthetic activations and no model weights.
CI reuses the existing H3 CPU conformance environment for this proof, and separately
builds the dependency/diagnostic wheels and describes the bounded output schema.

Owner: Codex /root/upload_performance, transferred from /root on 2026-09-07.
The original branch/PR114 and published 1.0.9 are preserved. GPU qualification
and publication of 1.0.10 require root's review of this candidate.
