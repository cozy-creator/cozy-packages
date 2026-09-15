# Runtime model-library dependency

H3 workflows import `H3Model`, `H3TurboBase` and `H3TurboLoRA` from `cozy_runtime.models.minimax_h3`. The package owns request validation, media policies and long-form orchestration; Runtime owns inference and its tokenizer/processor resources.

The published Runtime0.18.3 wheel contains these model classes. It does not contain the later Turbo Sol policy API or the Qwen position-cache cleanup that current package serving requires. Import availability alone is not sufficient to publish this migration. Qualification must use the exact content-addressed development wheel containing those changes; publication must wait for a reviewed worker-image/Runtime artifact delivering them through the existing release preflight. No new PyPI version is assumed here.

The package version stays at current1.15.2 during review. Its Runtime floor remains>=0.18.3; this is the current published class floor, not a claim that all0.18.3 artifacts contain the later changes. Do not publish or activate this draft until its final artifact requirement is settled.

Source9798fa33 and the earlier migration supplied the workflow used by the retained four-H100 PDD8 dense/Sol clips. That proves the core base-plus-adapter inference path, not every current long-form change. Current conformance must additionally preserve optional seeds, internal shot renderers, default Turbo bindings, overlapping assembly and retained-prefix recovery.

Prefix provenance includes actual Runtime H3 inference/resources in addition to workflow code and versions. Checked-in schedule JSON remains only for static request-step metadata and is byte-compared with Runtime resources.
