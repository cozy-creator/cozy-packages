# Fixed subjects for camera-cut stories

`long_form_cuts` generates one fixed image per named character or scene using
`paul/qwen-image/generate` before rendering any H3 shot. Character images use a
plain white background. Scene images contain the described location. The whole
request declares one to nine references; each shot explicitly selects its named
references. Each shot is an independent Ref2VA generation, with no first/last-frame
anchor and no references sampled from earlier generated video.

The parent is normal Python composition running in Runtime on the selected
machine. It awaits sequential Qwen calls, followed by sequential H3 calls. The
worker owns model placement, memory admission and eviction between packages.
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
