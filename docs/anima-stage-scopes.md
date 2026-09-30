# Anima scoped residency and explicit retry contracts

Anima previously held all four component scopes throughout its public render method. Ordinary module-level orchestration now keeps the same single upstream pipeline call while existing Model scopes cover encoding, conditioning, denoising and VAE blocks. Setup still admits all components honestly; Python-only FBC cleanup stays in finally even after admission/device failure. No loop, scheduler, generator, numerical mode or dependency other than the required Runtime floor is changed.

## Contracts in this candidate

The source now includes only the reviewed R19 Anima contracts needed for coherent Runtime residency/retry behavior:

- Cosmos transformer and text conditioner declare their inference forward state immutable.
- The text encoder declares purity only for default RoPE. Dynamic RoPE remains unclaimed because it can rewrite buffers.
- The Qwen VAE supplies save/restore of its named decoder/encoder cache-list bindings and index values by identity. It does not claim those caches are immutable or copy arbitrary tensor graphs.
- FBC explicitly owns its condition-specific cache manager/context, residual bindings, lazy hook traversal and parameter-index state. It claims only known installed hook/forward identities, composes existing claims, and restores the previous declarations when detached. Unknown forward replacements or unrelated hooks do not gain retry authority.

These are state/ownership promises, not numerical rewrites. The constructor's BF16 dtypes, VAE tiling, upstream blocks, serial guidance order, CFG default interval0.15–0.7 and FBC default0 remain unchanged. Public configs, tokenizer assets, model metadata and all other dependency requirements are unchanged. This candidate requires `cozy-runtime[media]>=0.18.93`, the reserved coherent Runtime release providing these author APIs. The package version remains unchanged pending coordinated release ownership.

The call-only block adapter preserves upstream metadata, state and implementation. It is installed on the actual constructor tree; execution through the public deepcopy property is tested. Unknown text2image topology refuses before mutation. Setup's returned pipeline object grants no new unscoped weight access. Existing root device authority still covers ordinary Python between stages.

## Validation and release gates

Fourteen guarded CPU tests use real tiny Qwen3, conditioner, Cosmos and Qwen VAE modules with the original pipeline/guider/scheduler. They compare all tensor state, image output, request/global RNG, full30step/60call CFG and public-default CFG behavior, FBC0/.075, scope lifetimes and poisoned cleanup. Additional tests exercise the actual constructor's default/dynamic RoPE claims, real VAE decode before/after Runtime state/RNG restoration, condition-specific FBC identity restoration and foreign hook/forward rejection. Cold root-configured package types, CI lint and generated-interface verification pass. These are not GPU or multi-GPU qualification.

The two private ordered/reversed GPU pairs motivate this candidate: the original1941/adapter1942 pair reported denoise103.8667→84.1902s, and the reversed pair70.869s scoped versus92.020s original, with exact RGB and reduced repeated weight movement. Those runs used private R19 Runtime and explicit contracts, not the unresolved public package dependency graph. Do not publish a performance guarantee from these pairs or infer that every resource envelope fits.

**Public lockfile blocker:** `anima/uv.lock` still records the prior public Runtime0.18.67. No source/version/hash was relabeled, no private wheel path was inserted, and unrelated dependencies were not re-resolved. Once the approved public Runtime0.18.93 artifact exists, refresh and verify the coherent lock and package wheel using ordinary release tooling. Until then, this clean source candidate is reviewable but not a release-qualified locked package.

Before publication, qualify the exact Runtime+contracts+scopes artifact set with matched repeats, unchanged inputs/defaults/dtypes/tile modes, state/RNG/output and OOM/replay evidence, ordinary CLI end-to-end runs and relevant low-memory/group cases. This draft does not establish release qualification; publication remains gated on that evidence. User release authorization is not being re-requested.

The application request/result interface is unchanged; generated metadata differs only in intended `component_use` maps. Python Model helper `render` becomes module orchestration `render_request`; repository warm/generate callers were migrated. External users of that helper need release review.

The pre-existing `packages/tracker-246-retry-current` worktree was read only. Its staged/unstaged Anima/SDXL and unrelated H3/script work was not changed or discarded. This owned branch integrates selected reviewed Anima declarations directly; private experiment PR365 remains unchanged.
