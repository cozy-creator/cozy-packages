# Isolated Anima stage-scope experiment

Private clone of frozen R19 Anima author/config/dependency source. Public Anima and immutable benchmark cohorts are untouched. This changes component lifetime boundaries only; it is not a GPU,1GiB-fit, performance or cross-engine numerical qualification.

`render_request` is ordinary package orchestration, not a public Model method with implicit ALL scope. `prepare_request` has an honest all-component scope for actual transformer-device lookup, request generator creation, upstream workflow construction/registration and guider/FBC setup. That scope closes before the original single `pipeline(...)` call. Existing upstream blocks are call-adapted to current `uses_components` methods:

| Original block | Components held |
|---|---|
| text_encoder | text_encoder |
| denoise.text_conditioning | text_conditioner, transformer |
| denoise.input / prepare_latents / set_timesteps / denoise | transformer |
| decode.decode | vae |
| decode.postprocess / phase announcement | no model method; output/state only |

The adapter changes only the block instance's class to a call-only subclass of its original class. Its inherited metadata/properties, original instance fields, loop/sub-block objects and upstream implementation remain intact. The original class's bound `__call__` is invoked under the corresponding Model method. No numerical loop, branch, PipelineState or scheduler is copied/reimplemented. It is installed on the actual blocks passed to the pipeline constructor, not the throwaway `pipeline.blocks` deepcopy. The actual deepcopy route is tested too. Unknown text2image topology refuses before any rewrite; this is a bounded private adapter, not Runtime source prediction.

Upstream ModularPipeline.__call__ sets state/default kwargs, calls `_blocks` once under no_grad, then extracts requested output. Component registration is inside setup; the package's original `_execution_device` override captures a real device. Encoding reads text-encoder dtype; conditioning reads conditioner/transformer dtype; denoise preparation reads transformer dtype; decoding reads VAE dtype/config and actual latent device. Those metadata reads remain inside the matching scopes. No fake Tensor/device metadata or Runtime policy/hint is introduced. Objects returned from setup carry real components, but subsequent weight/metadata operations remain behind the reviewed block scopes; no new permission for arbitrary out-of-scope access is implied.

Warm calls the same external orchestrator, with the original512px/one-step request unchanged. Tests observe that authored warm request without executing a downsized substitute. FBC installation, condition-specific cache behavior and restore code are unchanged. Cleanup directly restores only Python hook/forward/cache bindings in the original finally path. It performs no weight/device access or failable scope admission, so a poisoned planner cannot mask the original exception or prevent FBC restoration. The request generator, CFG interval/scale, two sequential condition calls, dtypes, attention implementation, tile256/stride192 and output postprocessing are unchanged.

Guarded CPU proof uses real tiny Qwen3, AnimaTextConditioner, two-layer Cosmos and Qwen VAE modules, a real tokenizer, original scheduler/guider and original upstream pipeline driver. For30steps, the transformer executes60 real calls. Scoped/unscoped executions compare every tensor in PipelineState, output image and final generator state exactly. Tests also cover FBC0.075, actual Model harness scope records, real Planner/_Scopes with VAE-only live scope, execution through the public deepcopy property, failure cleanup and unknown topology. These float32 CPU fixtures do not establish GPU kernel equivalence, released physical VRAM, group/remote execution, or fit of the unchanged1GiB request.

FROZEN-SOURCE.json records the original cloned bytes. Configs/tokenizers/operations.py remain byte-identical; pyproject/lock change only the private project identity, and metadata/package-interface.json is regenerated from this experiment. Its locked private R19 Runtime wheel paths are local experiment dependencies, not publication metadata. A GPU owner must review source and qualify a distinct immutable package candidate before any benchmark uses it.
