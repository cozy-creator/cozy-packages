# Private client scripts

Run a Python file directly with `cozy run ./path/to/script.py --rental-only`.
The client captures the script and its editable dependencies, then executes the
ordinary `main()` function on the private worker. These files are not published
packages and have no `package.toml`, App registry, or request/result classes.
Runtime 0.5.2 or the matching source candidate is required for typed model/service
parameters; the dependency is captured with the script.

- `h3_checkpoint_repair.py` is the completed, guarded repair recipe for exactly
  four historical H3 roots. A corrected checkpoint is refused before mutation.
  Its `[tool.cozy.weights]` metadata declares the finite output budget; `main`
  receives the existing Model and WeightsSink capabilities and returns a real
  ModelArtifact. The script performs no automatic upload or release mutation.
- `h3_vae_roundtrip.py` generates a fixed reference image, uses the existing H3
  VAE encode/decode methods, stages DiT between them, and records resident tensor
  hashes and Cozy Eval grid observations. It returns a video plus source and
  reconstruction PNGs. Change `FRAMES` or `CYCLE` in the file to change the probe.

Supply an exact reviewed model with the ordinary `model.source=<ref>` (repair)
or `model.model=<ref>` (VAE) term. Optional PEP 723 `[tool.cozy.models]` entries
can set these defaults; explicit CLI terms take precedence. Imports and model
classes come from the captured library environment. Script-local classes and
annotation expressions with function calls are not capability declarations.

The VAE example captures the adjacent editable `minimax-h3` library through
`tool.uv.sources`. Its `h3_diagnostics`, `h3_activation_trace`, and
`h3_resident_samples` modules share H3's existing inference and component scopes.
The activation helpers remain ordinary library functions: observers wrap the
original denoise callback and never replace the sampler. Historical fingerprint
expectations identify their original checkpoint and are observations, not universal
requirements for later checkpoints.

Completed memoized operations and independently retained partial work can be reused
by a new script on the same worker. A script itself starts from `main()` each time;
plain Python lines are not replayed or skipped by a workflow engine.
