# H3 preparation from a client script

`h3_tables.operations.precompute_adaln` is an ordinary async Python helper.
It composes managed operations alongside `assemble_full` and `quantize` in an
unpublished client script:

```python
from h3_tables.operations import assemble_full, precompute_adaln, quantize

async def main(ctx):
    # dits and shared are ModelArtifacts returned by the source/conversion operations.
    full = await assemble_full(dits=dits, shared=shared)
    pruned = await precompute_adaln(model=full, timesteps=50)
    fp8 = await quantize(source=pruned, encoding="fp8-rowwise/1")
    mxfp8 = await quantize(source=pruned, encoding="mxfp8/1")
```

This client-script path requires Runtime 0.12 or newer for its source-preserving
caller overlay and native attachments that add no tensor bytes.

The current captured plan is the approved union of 30, 40 and 50 denoising steps.
`timesteps` selects a supported step count; the retained BF16 bank supports all
three. Other schedules refuse until their package plan and serving contract are
qualified. The argument does not silently invent a new scheduler.

`precompute_adaln` runs in the calling script; it needs no `ctx` argument or
separate job invocation. It invokes two inexpensive generating-weight projections,
two independently memoized table computations and one native metadata attachment.
Runtime's caller overlay preserves this helper and the package's plan assets;
each leaf operation runs with its own managed services and cancellation checks.
Each table computation checkpoints completed block tables and final normalization;
a retry skips their projections and source-weight reads.

Projection content contains only the task's original generating weights and
required topology. Changes to unrelated body weights therefore preserve the
projection identity. Body quantization has no influence on table bytes; changing
generating weights, the plan, numerical contract or captured code invalidates
compatible reuse. Runtime owns the memo index; these operations introduce no
package cache or manual run identifier.

The returned model carries each generating projection's digest inside its existing
`cozy_h3` config stamp. Attaching a bank from other generating weights refuses even
when table shape and timestep plan match. An already-pruned current-plan model
with these bindings can be returned after validation. To replace older tables,
pass `generating_model=full`; the source must contain the original dynamic weights.
Older artifacts with no generating binding refuse this reuse path because the
removed weights cannot be recovered from a table's shape.

The native operation, small CPU math and generated-interface proofs are
`scripts/h3-adaln-{binding,resume,interface}-proof.py`. They do not establish
production GPU numerical quality or a paid remote pipeline qualification.
The interface driver takes `--implementation-wheel` pointing to the built tools
wheel and verifies its real caller overlay before exercising the helper's leaf calls.
