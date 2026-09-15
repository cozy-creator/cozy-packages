# One-forward H3 scaling profile

This private ordinary serving package binds the real FP8 H3 base and independent
PDD adapter. It runs the existing eight-step pipeline through one selected full
forward, then returns per-rank measurements and compressed Chrome traces through
normal output delivery. It does not render a clip or change production kernels.

Default: seed7101, the existing15-second courtyard fight, untimed step0 followed
by profiled step1, Sol dense8 policy. Stage independent FA3 and Sol packages with
`scripts/prepare-h3-scaling-profile.py`. The explicit source/hash arguments and
embedded Runtime guard bind each capture to a reviewed SDK. First target is the
same d140/39ee four-H100-NVL cohort that produced the measured slow run. A second
capture may use a separately reviewed SDK containing only the warm-residency fix;
record its exact manifest and hardware instead of silently accepting any SDK.
`expected_degree` supports1/2/4 with this same implementation. Do not substitute
H200 or B200 when four-H100 stock is absent.

The diagnostic model installs hooks on every replica at load time. A small
request marker crosses the existing mirrored call in `attention_kwargs`; a
prepend pre-hook removes it before Diffusers sees it. The marker names the target
zero-based evaluation and an attempt-spool output prefix. Only that forward
starts a profiler. On the leader, the stop exception occurs in the sampler's
completed-step callback, after component return and follower acknowledgements;
it never interrupts a collective and is caught inside the entrypoint.

Each rank records local Q/K/V shapes and post-exchange shapes. Shape checks use
sizes returned by the existing gather, requiring56/world heads and explaining any
trailing rank padding. No extra collective is used to assert shapes. The trace
labels QKV/output exchange setup and waits, the existing size-gather backend,
collective enqueue, actual FA3/Sol calls, every post-fusion linear and attention
module, PDD LoRA accumulation and Turbo final output heads. This includes the
complete50-block forward through final projections; it can expose global output
head replication as well as shard-local GEMMs. Kernel launch counts, copies/casts,
NCCL activity and CUDA self-time are available from profiler events.

CPU call spans, CUDA kernels and waits overlap. Do not sum them into latency.
Per-rank forward wall time excludes trace export; leader step time includes
mirrored argument transport and export, so it is diagnostic and cannot replace
the existing unprofiled video benchmark. No per-operation CUDA synchronize calls
are added. Profiler stop itself may synchronize. `record_shapes`, memory profiling
and Python stacks are disabled to limit perturbation. Tensor shapes are recorded
by the narrow wrappers without copying tensor payloads.

Per-rank event/self-time summaries are written before trace export. Each raw trace
is bounded at256MiB for delivery; an oversized trace is omitted with an explicit
status in measurements. The compressed archive is bounded at256MiB, and JSON at
16MiB by the output contract. Ordinary returned FileAssets establish custody;
ephemeral spool files alone are not completion. Keep capture logs outside source.

CPU proof uses actual tiny H3 forwards with two/four Gloo ranks and checks that
instrumented and subsequent uninstrumented outputs equal the baseline at zero
tolerance, exported trace files exist, and nine QKV exchanges carry the expected
head shards. It uses28 divisible rows because upstream Gloo final-output gather
refuses uneven29-row tensors; native NCCL uneven full-document proof remains a
GPU gate. CPU profiler controls cannot establish CUDA timing accuracy.
