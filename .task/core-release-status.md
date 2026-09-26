# Local core release cut

Local-only publication target: http://127.0.0.1:8819. Do not mutate the production Hub or installed editable aliases.

- Qwen Image2.1 0.2.0: generation and reference editing, Runtime>=0.18.30.
- H3 1.18.1: current six-section/reference/dialogue/prefetch source, counted GPU ladders, Turbo base/LoRA defaults, Qwen>=0.2.0, TensorFS>=0.3.55.
- SDXL2.3.17 and H3tools2.12.10: explicit memo dependency contracts, Runtime>=0.18.30, TensorFS>=0.3.55.

Portable locks for packages without Hub dependencies will resolve the public Runtime0.18.31 release. Local H3 publication uses an owned source copy with the selected Hub index and exact local lock. The canonical production index stays authored in this repository; its Qwen0.2.0 lock cannot be resolved until that release is separately authorized and published on production. This is a separate production release gate, not a reason to block the requested local publication.
