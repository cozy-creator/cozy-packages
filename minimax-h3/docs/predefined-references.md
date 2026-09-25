# Fixed subjects for camera-cut stories

`long_form_cuts` generates one fixed image per named character or scene using
`paul/qwen-image-2/generate_image` before rendering any H3 shot. Character images use a
plain white background. Scene images contain the described location. The whole
request declares one to nine references; each shot explicitly selects its named
references. Each shot is an independent Ref2VA generation, with no first/last-frame
anchor and no references sampled from earlier generated video.

The parent is normal Python composition running in Runtime on the selected
machine without reserving GPUs. It submits the bounded reference set concurrently,
waits for every reference, then runs sequential H3 shots. Results remain associated
with definition order even if images finish out of order; a failed or canceled
reference cancels and drains its siblings before the parent returns. Runtime owns
GPU placement and concurrency: degree-one Qwen calls can use separate GPU replicas,
then H3 can use its supported multi-GPU group. Package coroutines alone do not prove
physical GPU parallelism; that requires the scheduler cohort and a rented proof.
Installing either package does not pull weights; explicit model preparation or
inference preparation supplies them. No Creator process runs on the worker.

Each selected image defines an independent `<Subject N>` describing a reusable
character or environment, linked to its `<Picture N>`. Retention analysis specifies
identity and appearance or environment features, while the shot supplies its own
composition and movement. `{name}` in authored prompt text becomes the associated
subject label. Missing, duplicate and unknown names are rejected before generation.
The still is never described as a starting frame or a target video composition.

The existing `long_form` continuous-shot API remains available for intentional
first-frame continuation. This is an explicit input-contract change to
`long_form_cuts`: generated-shot history is removed, not silently ignored.

Final outputs remain only the assembled MP4 and last frame. Fixed images and
individual shot outputs remain retained intermediate child results. Model quality
needs a real matched render; conformance only establishes routing, input validation,
reference custody and output assembly.

Reference: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt

## Qualification and dependencies

The consumer declares `qwen-image-2>=0.1.0` and explicitly maps it to the
`tensorhub-paul` index at `https://tensorhub.com/v1/index/paul/simple/`.
There is no PyPI or local-checkout fallback. A selected development Hub must be
applied consistently to index resolution, lock verification and publication;
changing the package namespace or weakening wheel hashes is not an override.
This qualification lock honestly records the local Hub at `127.0.0.1:8819`.
For local qualification, resolve in an owned project copy with only the named
index's URL changed to the local Hub, retaining `explicit = true`. Run ordinary
`uv lock` in that copy, review the resulting lock, then retain it with the canonical
source project. Do not use `uv lock --index name=url`: uv treats that replacement
as an ordinary index and drops its explicit-only scope. A locked export may use
the selected-Hub override because it cannot change the already-reviewed resolution.

It is not a production-index lock. Select the matching Hub for local publication;
regenerate and review the lock against production when that package exists there.

H3 1.17.0 requires Runtime 0.18.24 or newer for the qualified managed-call,
media-decoder and recovery cohort. Creator captures the dependency's immutable
implementation and generated caller interface. The source implementation takes
injected execution arguments; its installed caller exposes the flat
`prompt`, `aspect_ratio`, `megapixels`, `steps`, `seed` and `background` parameters used here.

`scripts/h3-cuts-proof.py` compiles that caller from the actual qwen-image-2
package interface, then exercises Runtime's real broker, generated result types,
media codecs and custody checks with synthetic renderers. It checks reference
concurrent submission, reverse-completion identity/order, sibling cancellation, independent Ref2VA routing, ten cuts without a shot-count cap,
bounded assembly, two public assets, stable seeds, preflight refusal, cancellation
and partial delivery. This is a CPU composition proof, not Qwen/H3 inference,
package-index transport, or visual-quality qualification.

The existing `long_form` continues to use first-frame continuation and its own
previously defined shot bound. It is not routed through this reference generator.
