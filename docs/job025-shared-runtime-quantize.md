# job-025 — one shared Runtime quantization operation

Owner: `/root/private_worker_finish` (package policy/composition); Runtime implementation coordinated with `/root/source_native_finish`; root owns Creator built-in intake and final release. Packages worktree `~/cozy/.worktrees/packages/job025-shared-quantize`, branch `refactor/job025-shared-quantize`, base `10b371449a4cf6b793b82cd779cb69f47adbf87f`. Runtime origin audited `26f0811cef75ff26140acbd5c198e2d8f10e8723`.

The Sep12 user correction replaces the job-022 per-family managed cache boundaries. The only memoized quantization operation belongs to cozy-runtime. Family packages own deterministic policy data and ordinary Python convenience functions. Existing frozen family evidence remains historical; moving a decorator without isolating the operation identity is insufficient.

## Concrete API and implementation split

Public managed name: `cozy_runtime.derive.operations.quantize(source, plan, encoding, max_relative_frobenius=None) -> ModelArtifact`. Arguments are keyword-only under the existing invocable calling convention. `source` uses existing derive-only `QuantizationSource` binding; caller still passes a genuine ModelArtifact. Exactly one Runtime-owned invocable declaration with memoize=True. It returns only the stable artifact; original execution telemetry retains receipts, timing, and partial fidelity facts.

`QuantizationPlan` is a closed msgspec Struct (forbid_unknown_fields=True), owned by Runtime `derive.operations`:

```python
components: tuple[str, ...]
keys: tuple[str, ...] = ()
selected_schema_digest: str | None = None
source_order_digests: tuple[str, ...] = ()
output_precision: Literal["normalize-f32-to-bf16", "preserve"] = "normalize-f32-to-bf16"
```

Encoding and `max_relative_frobenius` are explicit operation options, not duplicated in plan. Empty keys selects existing dynamic block-aligned rank-2 float `.weight` candidates from the named components. Nonempty keys names exactly the shared key list required in every named component; no partial selection or fallback. It is allowed to leave unrelated weights unselected. The selected schema digest validates each selected tensor's exact component, key, logical dtype, shape and stored part name/dtype/shape. No Python callable, import symbol, arbitrary expression, or family name is interpreted by Runtime.

Canonical selected schema digest: `canonical_json.digest([[component, key, logical_dtype, list(shape), [[part.name, part.dtype, list(part.shape)] for part in parts]], ...])`, rows ordered by plan.components then plan.keys; dynamic selection uses the existing source order within component. Plain source validation requires exactly one `value` part matching logical dtype/shape before fingerprinting, so encoded/malformed sources cannot be accepted by a caller-supplied digest. Canonical source-order digest: `canonical_json.digest([[component,key], ...])` in full native construction order. The fingerprint includes all selected geometry, not merely selected key names.

SDXL plan: components `(unet,)`, ordinary source-derived plain rank-2 block-aligned `.weight` selection, normalize-f32-to-bf16. Anima is identical with `(transformer,)`. This retains current F16/BF16 behavior and normalization of unselected F32 tensors.

H3 plan: ordered components `(fl2va_dit, ref2va_dit)`, the 313 reviewed shared key names, the exact selected-schema digest computed from the existing BF16 tensor specs for both actual components, preserve precision, and the same two full/pruned source-order digests derived from family assets. The current `dit` numerical alias is flattened into actual components when deriving the schema fingerprint. Generic validation checks the granted source has all exact keys/dtypes/shapes and one matching plain value part; its complete native construction order must match an allowed digest. Runtime then preserves that source order verbatim. Every unselected tensor and all configs are inherited. Do not normalize H3's unselected F32s or invent a smaller H3 model.

The existing facade and `quantize_component_into` remain the numerical execution path. Extend planning/target construction, not encoding math. Selected data+scale groups retain current checkpoint and skip-read semantics. Keep the old synchronous facade and deployed multi-lane job APIs intact; remove only the three new managed family quantize registrations/decorators. Plain family wrappers remain only for existing consumers, and invoke the Runtime shared operation.

## Two real capability constraints to cover

1. Plan transport: the approved compact plan serializes to 13,582 B for H3, comfortably below the existing 48 KiB managed-call cap. The reviewed source plan is138458 B and pruned order asset231543 B, but only key names and cryptographic geometry/order identities cross the managed call. No FileAsset, Outputs parameter, metadata child, compression language or cap increase. Root approved canonical-order-only input: old `_quantization_order` accepted a reversed tensor enumeration and reordered it; that historical conformance expectation becomes a strict typed wrong-order refusal. Normal assemble_full/apply_adaln producers already emit exact full/pruned orders and are preserved.

   Verified order authority: `WeightsSource` documents tuple order as manifest construction order; `weights_sink.source_structure` iterates native header components/tensors; TensorFS `docs.rs::marshal_header` explicitly preserves stored insertion order. Therefore the order digest is over actual construction order, not sorted enumeration. Add a native header check as well as the pure structure controls.
2. Runtime built-in intake: cozy-runtime currently has no cozy.application entry point. `static_interface.Reader.reachable` skips trusted Runtime modules; Creator `install.HasInstalledApplications` excludes image-owned distributions. Thus adding an ordinary App/decorator alone does not create a usable shared target. Add an explicitly recognized Runtime-owned export/intake path, backed by the exact installed Runtime artifact/source and qualified numerical dependency identity. Never capture the caller/family dependency closure as the shared target or ship Runtime again as an arbitrary private wheel.

Shared memo identity must include source manifest, exact canonical plan bytes, encoding/options, Runtime operation implementation/native dependency identities and actual relevant numeric environment. Caller script edits and family code edits that leave plan bytes unchanged must hit. Plan changes must miss. Runtime/code or relevant numerical dependency changes must miss. Validate this on actual retained request/memo rows, not only a descriptor.

The shared output bound must support existing H3 `MAX_QUANTIZED_BYTES = 2 * MAX_OUTPUT_BYTES + MAX_PRUNED_BYTES`, 70,867,091,456 B (66 GiB +128 KiB), not silently inherit the 32 GiB single-DiT bound or the 16 GiB SDXL slot. Output admission remains bounded; derive the declared shared ceiling from the existing supported contract, with ordinary model grants and transaction custody unchanged.

## Required verification

- Model-free closed plan/schema tests: duplicate/unknown/missing fields, finite settings, rowwise/MX shape checks, missing/extra selected keys and wrong shape/dtype, exact full/pruned native order acceptance and wrong-order refusal, precision retention, no family-code callback.
- Four genuine tiny native SDXL/Anima cells through local Creator CLI, both encodings, compared against pre-refactor golden artifacts; H3 real exported negative cases remain negative. No tiny positive H3 claim (80.26 GB selected source; source census separately banked).
- Same-source same-plan second script and unrelated family edit hit the one Runtime target; changed plan/encoding/threshold and changed relevant Runtime identity miss. Show no family quantize job/cache target remains.
- Existing native interruption/group-reuse/partial custody controls, then actual managed interrupted-group proof using shared operation. Retain original producing evidence on a hit.
- Actual CLI capture/intake without manual install or publishing a composition; read-only --describe alone does not suffice. Run with the user's normal automatic source capture in an isolated owned home. Root alone controls paid runs.

## Preserved preceding evidence

PR190 merged `10b371449a4cf6b793b82cd779cb69f47adbf87f`, all checks green. Frozen Runtime14.2/TFS36 MIME/Tree client remains `outputs/se-042-sdxl-assessment-client/gpu-source-0142-ab999219a5/packages` and is not edited here.

H3 actual negative proof is in `outputs/job023-managed-refusals-20260912`: six table selection/compute/composition cases, plus both quantization encodings in parent `job-ff0f3aec074a8680f9c781dc` (45.643 s; facts fd95d8fc41d910ee5dad38463dfc9a1a2b6c6d85859f8b61c3c874f4af6f2f29,636 B). Quantizer children `job-408bd8cca9069e75e452bc83` / `job-d310be26ffb30c0d770e8d8d` refused quantization_source and produced no model. Frozen tools2.12.1 uses locked Runtime16.2/TFS38; parent control uses public16.5. Last normal down stopped the isolated daemon and zero rentals; five previous table-proof requests remain canceling. This is a recorded cleanup limitation, not a reader or quantizer defect finding, and no records were altered.
