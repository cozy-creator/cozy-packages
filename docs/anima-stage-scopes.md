# Anima stage scope correction

Anima previously held all four components throughout its public render method. Ordinary module-level orchestration now retains the same single upstream pipeline call while existing Model scopes cover the actual encoding, conditioning, denoising and VAE blocks. Setup still admits all components honestly; direct Python-only FBC cleanup remains in finally even after admission/device failure. No loop, scheduler, generator, numerical mode or dependency is changed.

This is a scope-only port from private experiment PR365. Public package defaults remain CFG interval0.15–0.7, FBC0, BF16 models and existing VAE tiling. Public Runtime/dependency requirements, package version, configs, tokenizers and FBC/retry behavior remain unchanged. The frozen R19 experiment also contained explicit purity/retry-state declarations absent from public Anima; those are intentionally not included here. Its single GPU A/B pair is motivation, not a promised public-package speedup or release qualification.

The call-only block adapter preserves upstream metadata, state and implementation. It is installed on the actual constructor block tree; deepcopy execution is tested. Unknown text2image topology refuses before mutation. A returned pipeline object does not grant unscoped weight access: inspected execution reads component metadata only inside the corresponding stage. Existing root device authority still covers ordinary Python between stages.

Source review, matched repeated inference and ordinary end-to-end qualification are required before public release. No merge or publication is authorized by this draft.

## Coherent release dependency

The single original1941/adapter1942 GPU pair ran the private R19 memory Runtime **and** frozen explicit model purity/retry-state declarations. Its denoise timing103.8667→84.1902s, movement110.592→83.325GB and exact RGB match do not qualify this scope-only public-source port independently.

The existing owned worktree `packages/tracker-246-retry-current` contains staged and unstaged Anima/SDXL contract work. This PR does not copy, modify, discard or publish it. Public master Runtime's author API does not yet supply those memory-feature declarations. The coordinator must finalize/review that contract work in a separate PR, coordinate its Runtime dependency, and qualify the exact combined Runtime+package-contract+scope cohort before a memory-package release. Preserve public dependencies and version in this PR rather than silently shipping private-wheel pins.

Combined qualification must include matched repeated original/scoped package runs, unchanged payloads/default CFG/FBC/dtypes/tile modes, output/state/RNG comparisons and OOM/replay accounting, ordinary CLI end-to-end execution, and relevant low-memory/group cases. The observed single pair is a useful candidate signal, not repeatable throughput proof. Scope-only source review may proceed independently; public merge/release remains the coordinator's separate decision.

The application entrypoint request/result schema is unchanged. Generated interface changes only the intended `component_use` method map. The Python Model helper `render` becomes module orchestration `render_request`; repository callers were warm and generate, both migrated. External Python consumers of that helper must be checked before release.
