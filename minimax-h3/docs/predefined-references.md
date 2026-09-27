# Fixed subjects for camera-cut stories

`long_form_cuts` uses a supplied `image` or generates it from the shared `description`
using `paul/qwen-image-2/generate_image` before rendering any H3 shot. Generated character images use a
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
subject label. Names are case-insensitive. Missing, duplicate and unknown names are rejected before generation.
The still is never described as a starting frame or a target video composition.

The `long_form` continuous-shot API uses fixed references and completed AV-tail
continuation. This is an explicit input-contract change to
`long_form_cuts`: generated-shot history is removed, not silently ignored.

Final outputs remain only the assembled MP4 and last frame. Fixed images and
individual shot outputs remain retained intermediate child results. Model quality
needs a real matched render; conformance only establishes routing, input validation,
reference custody and output assembly.

Reference: https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt

## Qualification and dependencies

H3 declares `qwen-image-2>=0.2.0` from the `tensorhub` index: the publishing
account's own package on the target Hub, with no PyPI or local-checkout fallback.
Creator writes that index for the command's Hub and account. `cozy package lock`
pins the version, and each publication rebinds the lock to its target account's
index without changing any version. Publish the locked qwen-image-2 release to a
target Hub before publishing H3 there.

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

The `long_form` API uses this same mixed reference resolver and completed AV context,
with no shot-count cap. See [LONG_FORM.md](../LONG_FORM.md) for JSON filenames,
`style`, the six-section prompt format and shared audio direction.
