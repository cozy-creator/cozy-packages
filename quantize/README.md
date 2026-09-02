# tensorhub/quantize

Installable reviewed model-derivation jobs:

- `tensorhub/quantize/fp8`
- `tensorhub/quantize/mxfp8`
- `tensorhub/quantize/sdxl-three-lane`

Publisher ownership is not project metadata. Creator derives it from the current authenticated
Tensorhub account; the official references in this document assume the `tensorhub` account.

Each job takes one exact TensorFS model Manifest. Runtime exposes only the granted Manifest's
bounded component/key/dtype/shape/role structure; it exposes no store, path, object reference, or
configuration bytes. The shared `cozy-jobs` wheel uses those exact facts to select block-aligned
rank-two float weights, reads their source roles through `WeightsSink`, inherits all unselected
ObjectRefs, and commits derived Manifests. FP8 and MXFP8 both derive directly from the caller's
source; neither can use the other quantization as its parent.

`sdxl-three-lane` binds the reviewed `civitai/sdxl/single-file/1` source profile. One ordinary job
normalizes every plain float tensor to real BF16, then emits row-wise FP8 and MXFP8 siblings with
the same BF16 floor for unencoded tensors. The rule is structural and shared across reviewed SDXL,
Pony, and Illustrious single-file variants; there are no version-specific code paths.

The package has no filesystem/store input, publication credential, or authored hardware floor.
Each callable declares only a maximum of newly produced model bytes. No callable authors a GPU or
VRAM guess, and no model family is inferred from a filename.
