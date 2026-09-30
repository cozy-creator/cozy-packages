# Private matched preview experiment

This branch is an experiment, excluded from the public H3 and Runtime releases.

The requested pair uses one exact prompt, seed 2768991793, the same character and
background PNG bytes, 10 delivered seconds, and the same Ref2VA Turbo 8-step plan:

- A: generate 960×480 completed AV latents; lift video latents to 1536×768 with
  the pinned LBH/Director learned spatial upscaler; decode through the H3 VAEs.
- B: generate 1536×768 completed AV latents; decode through the same H3 VAEs.

Both H3 arms use the same explicit four-GPU RTX PRO 6000 Blackwell placement.
The upscaler is a separate single-GPU stage. Check the actual GPU UUIDs and
attention backend receipts for both H3 arms before comparing timings.

Both canvases have a 2:1 aspect ratio. Run 1937 used 1344×768; this comparison is
not presented as a replay at that original canvas. H3 samples 243 frames for a
10-second request; both arms use the existing zero-prefix delivery plan to return
240 frames at 24 fps and 320,000 audio samples. A preserves the low-resolution audio
latents and temporal grid exactly. It does not re-denoise. Native B has the same
numeric seed, but its larger noise grid changes draws, including later audio RNG.

The upscaler is the pinned FP16 safetensors at HF
`LBH-123-AI/Minimax_h3_latent_Upscaler@3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`,
member `minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`.
Its 690,592,672 bytes have SHA256
`043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2`.
The 322 tensors total 345,280,216 parameters. The copied architecture/normalization
is from Director 63834b9f8561aa14a6e5289ac3d20dee78bb7d23; both upstream licenses
are included. The checkpoint model card declares Apache-2.0.

Existing Runtime source operations download the exact carrier and convert it with
`as-is/1`. A CPU child validates every tensor layout and uses the native source/writer
capabilities to derive FP32 upscaler weights and config. `Model`, `Loader`, and
`uses_components` own inference residency. No larger media-input caps, package HTTP,
manual CUDA weight loading, or new TensorFS source profile is needed.

The exact private Runtime wheel is selected through the local wheel source and an
exact development-version requirement. It includes public .92 plus the explicit H3
canvas and retained-CPU-latent decode changes from Runtime PR #1013, plus the
reviewed captured-call precedence repair and Qwen console-monitor timing fix. Recreate its
`.benchmark-wheels/` file from that branch's source-derived build receipt before
locking or capturing this project. Preserve its exact version and wheel hash as
pinned in `pyproject.toml` and `uv.lock`; do not relabel it as a public SDK.

CPU qualification covers upstream geometry, the entire checkpoint meta census,
bitwise small-network 1.6× parity with Director, managed upscaler scope, retained
latent/audio custody, actual Runtime writer RPC/TensorFS derivation, all eight
managed parent/child routes, and real 10-second MP4 encoding with exactly three
final assets. The GPU model execution in the broker fixture is synthetic; it is
not a video quality or GPU memory qualification.

Resume through ordinary `cozy` only on the user's next provided worker. Terrence
has ended, and no replacement rental is authorized. Before submitting any work:

- Verify Worker Runtime **0.18.93 or later** carries the captured-call precedence
  repair. `Calls.bindings` runs in the Worker; the private SDK alone cannot repair
  that boundary in a .92 Worker.
- Verify CLI **0.1.21 or later**, while preserving the exact repaired private SDK
  wheel and version already pinned by this project.
- Check the new worker's actual GPU SKU and count against the planned **four RTX
  PRO 6000 Blackwell GPUs**. Recheck availability and active work; do not assume
  Terrence's old inventory or UUIDs describe the next worker.
- Complete the neutral two-segment H3 qualification, including Qwen references,
  segment music and an indexed live MP4 revision. Then run this package's CPU
  `software` probe and verify its actual executor SDK before the matched pair.

Keep both arms on the same verified four-card placement. Do not interrupt other
work or reduce the requested canvas on admission failure. Record the actual GPU
UUIDs, admitted model manifests and attention choices in the comparison metadata.
Acquisition and loading are separate from handler stages; one ordered pair is not
a statistically controlled warm-speed measurement.

Only `preview-upscaled.mp4`, `native-768p.mp4`, and `comparison.json` belong in the
final `~/.cozy/outputs/` directory. Source checkpoints, latents, and diagnostics stay
in private custody. No public upscaler API or package publication is part of this test.

The private `software` CPU job reports the actual executor SDK/module and whether
explicit canvas support is present, without initializing CUDA. Run it through the
same captured package before the GPU pair and retain its ordinary run evidence.

Terrence was shut down by the user before either GPU qualification was submitted.
Preserve the neutral two-segment qualification and paired inputs for the next
user-provided worker; do not rent a replacement. Both final GPU receipts and the
ordinary remote software probe remain pending. The fixed four-card Blackwell
placement must be rechecked against that worker before resuming.

## Asirpa placement qualification

The user-provided next worker is Asirpa, rental `pr-15975cbca908e1b3ef8f`.
Its actual inventory is four NVIDIA H100 80GB HBM3 GPUs (81,559 MiB each).
Both H3 arms now select the same four-card H100 gang; all canvases, prompt, seed,
reference bytes, Turbo schedule and upscale checkpoint remain as qualified.
This is a placement change from the stopped Terrence plan, not a cross-machine
speed comparison. User requests 1983 and 1982 must finish before submission.
