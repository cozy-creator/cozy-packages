# Temporary full-H3 exchange diagnostic

This unpublished diagnostic preserves the generator body, base/PDD bindings and
Sol policy from helper commit `925acad921ec4e93a0e5850eb1c5fdeb1264f45c` and pairs
with H3 workflow `9798fa332736b4fe8fe8019f58bf4241c88a6438`. It requires the exact
reviewed Runtime source `d140609bf5fd07f8c67eaf7e87de433f5177f94f` (development wheel
SHA256 `39ee661cb41da8af215a9f4e8acec365f75bc2a3202557de3b0f11b58f7434db`). It is
not a supported public attention option and does not modify an installed SDK.

Full H3 failed on 2xH100 SXM before Sol executed, in the initial QKV all-to-all.
A tiny real-worker H3 fixture passed on the same GPUs with expandable allocations
and the same UID restrictions. This helper observes the remaining full-model case.

The only model change adds `load()` instrumentation. Runtime calls this on every
rank, unlike the leader-owned `warm()` method. Installation does not initialize
CUDA or change the model. Inside that process, it wraps the existing Sol
`_execute_parallel` function; Sol's original operand/layout checks still run, and
the original parallel function performs both exchanges and attention unchanged.
The first parallel call is the first post-fill warm exchange: backend qualification
has no parallel configuration and does not enter this seam.

Each rank emits one `h3a093.first_exchange` sequence to retained stderr:

- `entered`: rank, world, site, step, live length and Q/K/V shape, stride, dtype,
  device, storage offset and contiguity. No values or pointers are printed.
- `before_sync`: current device and allocator allocated/reserved plus driver
  free/total bytes. A failed CUDA metadata query emits `metadata_failed` with the
  measurements obtained so far and re-raises; missing values are not invented.
- `sync_failed` or `sync_passed`: explicit synchronization of the query device
  immediately before the unchanged parallel implementation. A failure here can
  expose preceding asynchronous CUDA work; it is not an exchange result.
- `exchange_path_failed` or `exchange_path_returned`: the original implementation's
  result. Its retained traceback distinguishes QKV, attention and output exchange.

The wrapper neither repairs device selection nor retries or swallows errors. It
logs only the first parallel call in each rank process, then directly delegates.
It changes scheduling through synchronization, so its latency is **not a benchmark**.
No allocator, isolation, communication or numerical policy is changed.

Stage a private capture using the pinned workflow and Runtime wheel through
`tool.uv.sources`, with default/dev groups disabled and image-owned CUDA/Torch.
Install under its separate `local/h3-sol-exchange-diagnostic` name with ordinary
Creator. Before any rental, verify import/schema, exact generator-body equality,
model bindings/degrees, dependency closure and captured revision. Submit only to
an explicitly owned 2xH100 SXM rental on image
`sha256:62d2b62cc4e72532cbf7e39e4931a8465d10e38777ef919b68640152e42e0d9c`, using the
same fifteen-second fight input, seed7101 and Sol dense8. Capture the entire stderr
and the outcome, even when warmup refuses before an inference attempt.
