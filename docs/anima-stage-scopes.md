# Anima stage scope correction

Anima previously held all four components throughout its public render method. Ordinary module-level orchestration now retains the same single upstream pipeline call while existing Model scopes cover the actual encoding, conditioning, denoising and VAE blocks. Setup still admits all components honestly; direct Python-only FBC cleanup remains in finally even after admission/device failure. No loop, scheduler, generator, numerical mode or dependency is changed.

This is a scope-only port from private experiment PR365. Public package defaults remain CFG interval0.15–0.7, FBC0, BF16 models and existing VAE tiling. Public Runtime/dependency requirements, package version, configs, tokenizers and FBC/retry behavior remain unchanged. The frozen R19 experiment also contained explicit purity/retry-state declarations absent from public Anima; those are intentionally not included here. Its single GPU A/B pair is motivation, not a promised public-package speedup or release qualification.

The call-only block adapter preserves upstream metadata, state and implementation. It is installed on the actual constructor block tree; deepcopy execution is tested. Unknown text2image topology refuses before mutation. A returned pipeline object does not grant unscoped weight access: inspected execution reads component metadata only inside the corresponding stage. Existing root device authority still covers ordinary Python between stages.

Source review, matched repeated inference and ordinary end-to-end qualification are required before public release. No merge or publication is authorized by this draft.
