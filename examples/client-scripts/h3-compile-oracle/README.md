This private experiment compares standard 30-step H3 FL2VA inference with eager
execution, regional compilation of the repeated DiT blocks (`mode=blocks`), or
compilation of the whole DiT callable (`mode=dit`). It keeps the serving wheel's
model code, checkpoint, precision, attention route, and existing Runtime fusions.
The purpose is to identify useful direct inference optimizations from compiler
lowering and CUDA traces. It does not enable compilation in the published model.

Prepare a separate project from the exact serving wheel under investigation:

```sh
uv run --no-project python \
  examples/client-scripts/h3-compile-oracle/prepare_h3_compile_oracle.py \
  /absolute/path/to/minimax_h3.whl /absolute/path/to/oracle/project
uv lock --project /absolute/path/to/oracle/project
cozy package install /absolute/path/to/oracle/project --editable --no-model-download
```

The preparer reuses the first-step diagnostic's wheel extraction. It copies all
serving members unchanged, retains dependency ranges, adds only the oracle
module, and sets `default-groups = []`. No new Runtime release is required.
If explicitly creating a local environment, use
`uv sync --locked --no-dev --no-default-groups --project /absolute/path/to/oracle/project`.
The neighboring `project.provenance.json` records source-wheel and helper hashes.
Freeze the generated project before submission; keep inputs, logs, and results
outside it so editable capture does not replace the executor during comparison.

Use one physical H100 on an existing owned rental, the ordinary default Cozy
home, and an immutable checkpoint. For the H3 1.14.3 standard qualification:

```sh
cozy run local/h3-compile-oracle/probe \
  'model.model=paul/minimax-h3#sha256:d64f250c556b28889fb0c900643bf2028fa92cad619d8cc4b716d6145ca7d275' \
  'prompt=A woman sings softly into a microphone on a warmly lit jazz stage while a pianist accompanies her. Her lips move naturally with the singing. A slow camera push-in, steady lighting, clear voice and piano.' \
  seed=7101 duration_s=15 mode=eager profile=true \
  --rental=OWNED_RENTAL --await --json --out=/absolute/path/to/results/eager-profile
```

Repeat that exact input and binding in this order, changing only the indicated
entrypoint/fields and output directory. Keep profiler runs separate from full
latency measurements:

| Arm | Entrypoint | Fields |
|---|---|---|
| Eager first-step profile | `probe` | `mode=eager profile=true` |
| Cold regional compilation | `probe` | `mode=blocks cold=true profile=false` |
| Full eager video | `run` | `mode=eager profile=false warmup_steps=2` |
| Full warm compiled video | `run` | `mode=blocks profile=false warmup_steps=2` |
| Warm compiled first-step profile | `probe` | `mode=blocks profile=true warmup_steps=2` |

`run` produces the ordinary video and continuation frame plus measurements and
compiler artifacts. `probe` deliberately stops after the first denoising step,
before video decoding; successful probe status is `first_step_only`. An outer
completed request can contain a probe with `status=error`: inspect its retained
measurements and traceback before calling compilation successful.

The first compiled call requires Runtime to impose its compiler caches before
executor startup. `cold=true` refuses a nonempty Inductor cache; it does not erase
or replace the sealed cache paths. Pre-existing Runtime Triton kernels remain
available. Record cache inventories, PID, versions, worker/GPU identity, graph
counts, and counters from the measurements. A warm run should reuse the same
executor and configuration with no new compiler graphs. Changing mode or
`autotune=true` creates a new compiler configuration; its first call can still
reuse disk caches and must not be labeled a fully cold boot.

An unchanged project does not guarantee that a worker retains its executor
between requests. Use `warmup_steps=2` for warm comparisons: within one request,
the oracle first runs two denoising steps without profiling or decoding, then
restarts the unchanged, seeded `fl2va` call on that same model and compiler.
Warmup is recorded separately under `warmup` and excluded from measured
generation and step times; CLI execution still includes it. Two steps cover
specializations first reached on the second scheduler step. Check warmup and
measured PIDs match and `graphs_before == graphs_after` during the measured pass;
additional captured graphs mean compilation still occurred. `warmup_steps`
accepts 0, 1, or 2; `cold=true` applies before warmup only.

Compilation uses `fullgraph=false`, `dynamic=false`, and no CUDA graphs. Graph
breaks are reported, so `mode=dit` does not imply complete DiT graph capture.
The graph records contain each backend callback's elapsed time and exact FX
source. PyTorch `compile_times` categories are nested and process cumulative:
do not add them together or treat later requests' totals as new compilation.
Compare `generation_seconds`, `denoise_seconds`, all 30 step times, and CLI
execution time separately. CLI execution also includes output finalization;
queue time and model preparation are not warm inference. Hash and decode the
returned video/audio before interpreting a speed difference as equivalent work.

Extract each `compiler_artifacts` gzip archive into a separate directory. Select
the exact profile basename named in that request's `measurements.profile`; a
warm cache archive may contain earlier traces too. Analyze the two equal
first-step profile windows with:

```sh
uv run --no-project python scripts/h3-compile-trace-summary.py \
  /absolute/path/to/eager/diagnostic/profile-TIMESTAMP.json \
  /absolute/path/to/compiled/diagnostic/profile-TIMESTAMP.json \
  --generated-source /absolute/path/to/compiled/inductor \
  --out /absolute/path/to/results/kernel-comparison
```

The JSON/Markdown comparison retains kernel counts and durations, CUDA launch
counts, device busy interval unions, and generated source identities. CPU
operator durations are inclusive and may overlap asynchronous GPU work. Static
generated call sites are not execution counts. Use actual CUDA traces to check
which kernels changed and how much total time they can explain; measure any
proposed direct optimization separately on full inference.
