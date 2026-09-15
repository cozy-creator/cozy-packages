# Runtime model-library dependency

H3 workflows import `H3Model`, `H3TurboBase` and `H3TurboLoRA` from `cozy_runtime.models.minimax_h3`. The package owns request validation, media policies and long-form orchestration; Runtime owns inference and its tokenizer/processor resources.

Published Runtime0.18.3 contains the model classes and Qwen position-cache cleanup. Published0.18.4 also contains the Sol Turbo policy API, so this migration declares>=0.18.4. These shipping artifacts were inspected directly; class availability and policy defaults are separate facts.

Runtime's later four-dense-step Turbo default is currently qualified with the exact content-addressed b0ccd805 development wheel. The published0.18.4 model still defaults to ten dense steps. Do not claim the new default is active for normal users until the chosen worker SDK delivers it. Package publication remains subject to the existing actual-worker preflight; no new PyPI release, shared install or activation is performed by this source change.

The package version stays at current1.15.2 during review. Source9798fa33 and the earlier migration supplied the workflow used by the retained four-H100 PDD8 dense/Sol clips. That proves the core base-plus-adapter inference path, not every current long-form change. Current conformance additionally preserves optional seeds, internal shot renderers, default Turbo bindings, overlapping assembly and retained-prefix recovery.

Prefix provenance includes actual Runtime H3 inference/resources in addition to workflow code and versions. Checked-in schedule JSON remains only for static request-step metadata and is byte-compared with Runtime resources.
