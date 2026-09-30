# Stable SDXL decoder representation experiment

Frozen R19 SDXL converts VAE weights to the execution dtype and back on every
preflight and final decode. Each conversion invalidates Runtime's working-memory
profile. An unmeasured decode then conservatively vacates sibling weights, including
the UNet needed by the next image.

This private experiment selects the same execution dtype as before: native CUDA
BF16 where supported, otherwise FP32, only when the VAE requests upcasting. It keeps
that representation after successful conversion. A package-owned flag is set only
after the complete conversion returns; the first parameter's dtype cannot establish
success after a partial allocation failure. Runtime continues to own placement,
accounting, reconstruction conversions and retry. A fresh pipeline resets the flag.

The decoder operations, latent conversion, scaling, tiling selection/freeze, scheduler,
steps, guidance, checkpoint and request are unchanged. The experiment introduces no
new memory estimate or authority. It may retain FP32 backing between decodes on devices
without native BF16; the existing policy must charge that real size and page it normally.

CPU checks use actual tiny Diffusers VAE decoding against the frozen prior helper,
Runtime conversion epochs, partial conversion failure, decode failure and RNG state.
They do not establish full-model GPU equality or a speedup. Require complete grouped
inference, image checks, transfer volumes and thermal observations before promotion.

This is scoped to the package's owned VAE: it has no other dtype writer, VAE encode
consumer, or component replacement between renders. It does not promise identical
caller-visible dtype after decode or handle arbitrary external mutation. Ordinary
BF16 checkpoint values may roundtrip through FP16 unchanged, but values near FP16
overflow need not; full checkpoint/image checks are required rather than a universal
claim about casting algebra. Disk replay is exercised with actual Runtime tiers and
a canonical CPU store; native TensorFS and GPU BF16 dispatch remain inference gates.
