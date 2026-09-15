# H3 attention qualification

`main.py` runs this diagnostic through ordinary `cozy run` on an explicitly owned
single-GPU rental. It imports H3 inference from the paired Runtime built-in model
artifact through the workflow package; it does not copy a transformer or kernel.
Published Runtime 0.18.2 lacks these APIs. Use the reviewed development wheel in
both capture and worker; a successful static describe alone is not execution proof.
The helper modules are an ordinary Python package declared in `pyproject.toml`;
`main.py` depends on it explicitly. Creator does not implicitly capture adjacent
Python files. For qualification, stage a copy of the script whose `tool.uv.sources`
pins the exact Runtime development wheel and the local helper/workflow packages.
Selecting the host tool through PATH alone does not pin capture dependencies.

Edit the ordinary script's `REQUEST` before capture: `prompt`, `duration_s` (15),
`capture_step`, `capture_block`, `backends`, `repeats` and `reference_backend`.
The package-style `h3_attention_oracle:app/probe` entrypoint also accepts these as
request fields. For example, compare
`["fa3_bf16", "sage2_sm90", "kitchen-int8"]` on H100, or
`["sdpa", "flashinfer-bf16-fp8"]` with `reference_backend="sdpa"` on Blackwell.
The model binding is the ordinary `model.model=paul/minimax-h3` checkpoint option.

The script captures actual Q/K/V at the selected module and step, computes the
same reference for all candidates, and includes required preprocessing in timings.
Construction timing includes Runtime's small native smoke and any JIT it triggers;
first-call timing then measures the first call at the full captured geometry.
Outputs include complete-call wall/GPU times, FP32 sampled-row error, full-output
reference error, repeatability, input hashes, profiles and effective backend options.
These measurements describe a kernel call, not an end-to-end video speedup.

For Sol use the complete sequence, `capture_step >= 10` and `capture_block >= 2`
to exercise the current sparse H3 policy. Earlier steps/blocks intentionally report
dense calls. The diagnostic retains the model's actual live-token/prefix metadata;
it does not invent conditioning ranges or silently change the warmup policy.
`last_effective_calls` distinguishes sparse calls from protected dense query work.

Version 2 result documents name `against_reference` explicitly because Blackwell
comparisons need not use the H100-only FA3 artifact. Reference trajectories, checkpoint
identity and captured input hashes must match before comparing separate experiments.
Native-kernel tests and matched full videos remain separate qualification gates.

The private App's `generate` entrypoint renders a complete 30-step video through the
ordinary H3 workflow. It accepts `prompt`, `seed` and `duration_s`; choose attention
through the normal scoped request override. Run identical inputs and checkpoint
with `--attention-kernel=model/fl2va_dit=flash-attn3` and
`--attention-kernel=model/fl2va_dit=sol-attn` for a visual pair. Record the resolved backend
and effective dense/sparse counts from each attempt. Unlike `probe`, `generate`
does not replace processors to force a capture trajectory or stop denoising early.
