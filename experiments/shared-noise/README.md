# Private shared raw-noise quality input experiment

This draft prepares two canonical CPUF32 inputs and isolated adapters. It is not a
public seed change, performance cell, neural quality metric or cross-engine bitwise
parity claim. Old benchmark files/defaults remain unchanged. TensorFS/Varena adoption
is unrelated and remains design-only.

Controls are exact SDXL stable-VAE37ff2b8 and Anima scope0d76083. Project identities
change privately; assets/dependency pins and every non-injection authored AST remain
exact. Current-builder interfaces equal each control. Inherited R19 private pins are
preserved inputs, not a claim of a prepared R20 installation; coordinator must freeze
matched current cohort pins before registration.

## Canonical input

`generate_noise.py` creates CPU contiguous little-endian float32 bytes with an owned
`torch.Generator('cpu').manual_seed(seed)`, without touching global RNG. The actual
artifacts live under `analysis-shared-initial-noise/artifacts/` in the benchmark output
root; manifest records shape/axis order/byte count/hash, Torch build and generator states.
They are private input artifacts, never model weights or checkpoint substitutions.

- SDXL1005: NCHW[1,4,128,128],262,144bytes.
- Anima1006: NCTHW[1,16,1,128,128],1,048,576bytes.

All adapters read those same files and reject hash/shape/seed mismatch, without a
random fallback. No global torch.randn or module-function monkeypatch is installed.
The trusted private absolute path is deliberate for this host-only draft; ordinary
remote deployment requires separately declared/granted inputs, not ambient path access.

SDXL replaces only its random draw with artifact→existingF16/device conversion, before
unchanged init_noise_sigma multiplication. The existing request generator is still
created but no longer consumes that initial draw. Anima's real generate path passes
CPUF32 `initial_noise` through the ordinary helper to upstream pipeline `latents=`;
the existing prepare_latents path handles the original F32/device move. Its lifecycle
warm call passesNone and retains the original behavior.

Comfy uses an explicitly named private KSampler node. Its local common_ksampler copy
is source-identical except the prepare_noise provider call; the ordinary sampler,
callback, masks and output path remain. Public KSampler/torch RNG functions are not
replaced. Node activation would require a separately reviewed explicit custom-node
allowlist; it has not occurred. After Comfy fix_empty_latent_channels, expected Anima
shape is accepted as5D directly or4D only by removing canonical singleton temporal
axis2. Same element count does not authorize other reshape/transposition.

## RNG side effects and numerical limits

This is a deliberate noise-input intervention. Comfy's original prepare_noise calls
torch.manual_seed(seed), resetting global generators, then consumes CPU draws. The
artifact loader does neither. Cozy's request CUDA draw is likewise bypassed, changing
that generator's post-draw state. CPU tests explicitly demonstrate this difference;
they do not claim original global/request RNG states are preserved.

The selected deterministic Euler path has no churn and the selected FlowMatch lane
must keep stochastic_samplingfalse. No authored later draw is inferred safe merely
from a generic scheduler name; freeze and inspect actual configs/options. Hidden
custom stochastic nodes/HiDiffusion/FBC or a changed scheduler are outside this draft.
Do not silently consume/discard random draws to make a test appear equivalent.

Raw canonical artifact equality is proved. **Actual first denoiser inputs are not yet
proved equal**: Cozy SDXL rounds theF32 input toF16 before scaling, while Comfy retains
F32 sampling state; scheduler scaling/grids/conditioning also differ. The CPU test
asserts that thisF16 roundtrip differs from sourceF32. This is intentional disclosure,
not a reason to change precision/math opportunistically. Shared quantized values would
be a separately authorized experiment.

## Tests and remaining pre-launch gate

Eight guarded CPU tests cover canonicalTorch regeneration/roundtrip, dtype conversion,
explicit axes, badseed/shape/indexedbatch/corruption refusal, actual Diffusers Anima
prepare_latents, Comfy source AST noise-only delta, globalRNG sideeffects and all frozen
asset/non-injection AST equality. No full model construction orGPU. Current static
interfaces compare byte-equal. Source runtime/device behavior still needs review.

First proposed trial, only after frozen review: one fullSDXL1005 and one fullAnima1006
perengine,1024²,20/30steps,CFG7/4.5,original prompt/negative/checkpoints,AnimaCFG0–1/FBC0,
SDXLHiDiffusionoff,stableVAE,8GBuncapped and existinghost/CPUcaps. Do not run a smaller
proxy or score diagnostic time as production performance. Keep old controls separately.

Before registration/install/GPU, freeze coherentRuntime/package/node/artifact/config
identities and a bounded observer recipe for actual noise-after-cast/scaling, first
model input, conditioning and schedule values. This input-only draft does not install
that observer or claim those gates passed. No post-adopt callable mutation or global
patch is allowed; any needed class delegate must be independently reviewed before
construction. If the supported boundary cannot observe a value safely, report it
unknown and hold equivalence attribution. Record differences instead of modifying
precision/sampler/steps until the hashes match.

On later approved outputs, compare visual composition/prompt adherence/artifacts using
actual originals and human review. Shared noise removes one confound, not all. Stop
and retain unexpected failures; no fallback mode or public release is authorized here.
