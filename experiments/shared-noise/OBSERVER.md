# Bounded observer implementation (CPU qualification only)

This extends the separately preserved input-only freeze1b2dc456. No registration,
custom-node activation, installation orGPU run has occurred. Canonical noise files
and public seed behavior remain unchanged. The observers are diagnostic; their D2H
synchronization and CPU work can perturb scheduling and numerical workspace selection.

Cozy SDXL's signature-preserving delegate is defined on its private UNet class before
construction/adoption. The existing first denoise call is enclosed by external
ContextVar bookkeeping. Anima's private Cosmos subclass delegates the original forward;
one transparent AnimaLoopDenoiser block delegate records first-step state/schedule and
enters the observation group. No denoising loop is rewritten. Repeated upstream workflow
objects are recognized to avoid stacking the same observer. The original stage-scoped
Model wrappers and lifecycle warm behavior remain. Context-free warm runs are not recorded.

Comfy's private node creates ordinary ModelPatcher.clone() without force_deepcopy or
disable_dynamic and adds keyed official SAMPLER_SAMPLE/PREDICT_NOISE/DIFFUSION_MODEL
wrappers. Original wrapper lists and weight pointers remain unchanged. The sampler
wrapper records supplied noise/actualsigmas; PREDICT_NOISE records SAMPLER STATE before
BaseModel.calculate_input; DIFFUSION_MODEL records actual after-scaling/cast network
arguments and post-Anima-conditioner context. Received WrapperExecutor continuation
is called exactly once. Raw network prediction is labelled epsilon(SDXL) or
flow_velocity(Anima), not reconstructed denoised sample. No global class/function patch
or post-adoption module.forward replacement occurs.

The first-step group's actual positive/negative branch mapping is retained. Cozy SDXL
uses its authored negative→positive batch order. Anima compares actual post-conditioner
context bytes/dtype to first-steppositive/negative tensors; equal/ambiguous contexts
refuse rather than invent identity. Comfy uses actual cond_or_uncond chunk metadata and
requires the fixed batch1-per-branch geometry. Each branch's input/context/timestep and
raw prediction are separately hashed without permuting execution. Failed/retried or
repeated first-step groups cannot silently become a successful single-call proof.

Noise-provider return records carry the verified canonical source hash/shape and the
actual returned cast/axes. Comfy's allowed temporal squeeze is labelled explicitly.
Tiny schedule/timestep tensors retain values as well as hashes. Full bounded typed
scheduler/model configs distinguish lists/tuples/floathex and supported Torch dtype
enums; unsupported objects are refused, not stringified. No semantic normalization is
used to erase scheduler/model architecture differences in this experiment.

Each tensor readback is at most4MiB: two branches×512tokens×2048channels×two bytes;
initialF32latent is at most1MiB. Larger/unsupported inputs refuse the diagnostic. CUDA noncontiguous and lazy conjugate/negative views are rejected before any
copy/accounting/allocator query. Torch Copy.cu can otherwise allocate an internal CUDA
packing temporary even when .contiguous() appears only after .to(cpu). Only bounded
CPU-source packing remains supported, at most two temporary CPU buffers; no pinned
pool is created. A later real unsupported CUDA context must refuse this diagnostic,
not be packed onGPU or silently coerced. The complete record allows at most64MiB cumulative copied bytes and24events,
not64MiB retained tensors. Every copy has a unique ledger ordinal; the consumer requires
all ordinals/bytes and unchanged observer-only allocator/OOM/retry counters. CPU/global,
device and explicit request generator states are read only, never reset to equalize engines.

Consumer gates require complete provider/firststate/schedule/config/settings/nativeTID,
source identity, successful mapped branches, actualinput/outputdtype/shape, exactcopy
ledger and read-only allocator witnesses. Missing modes, unknownbranch, unsupported
schema, failed call/retry, wrongsource or partialcopy reject qualification. Comparison
never casts, rescales or reshapes to make actual values equal. Even valid common fields
do not prove fullnetworkargument equivalence: SDXL pooled/time-ID versus Comfy y schemas
and differing scheduler parameterizations remain explicit. The report deliberately keeps
full_network_argument_equivalence_established=false.

CPU evidence includes actualtinyDiffusersSDXL underRuntime managed wrapping, actualtiny
upstreamAnima pipeline andModelscopes, original exception identity/contextcleanup, global
and explicitRNG/output equality with observation off/on, actualofficialComfyWrapperExecutor,
actualModelPatcherclone sameweight/dynamic-policy proof, fullmetaSDXL/Cosmos unitpartition
comparison, rawbitbounds/ambiguousbranch/consumernegativecases. This does not establish
GPUcopy neutrality, actualsamefirstdenoiser inputs or cross-engine quality equivalence.

Before anyGPU: root+independent review exactfrozen source/consumer, coherent private
currentRuntime pins/locks and explicitcustomnodeallowlist, then separately authorized
fullSDXL1005/Anima1006 pair perengine at8GB. Preserve every original image/error. No
weightreadback, nativebackendlogs, compile/precision/sampler rescue or extra cells.

The prior17d1468/cd9c22a2 observer and39ace74d assembled cohort were unlaunched and
remain preserved as superseded evidence. Equal after-copy allocator counters were
insufficient to exclude a freed CUDA temporary. The corrected guard relies on the
reviewed contiguous same-dtype Torch copy path; it is not a physicalGPUneutrality test.
Coherent cohort/plugin snapshots must be regenerated only after exact guard review.
