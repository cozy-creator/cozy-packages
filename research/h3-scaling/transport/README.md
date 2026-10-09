# H3 direct-GPU versus host-memory transport

This research recipe changes NCCL transport for an otherwise unchanged four-H100
Turbo request: 15 seconds, 362 frames, eight steps (four dense plus four Sol).
It was exercised with Runtime 0.22.0, TensorFS 0.6.1, TensorD 0.4.0 and NCCL 2.30.7.
It does not change the installed Runtime, kernels, checkpoints or inference math.

Prepare the ordinary exact-four-GPU package with the parent `prepare.py` first.
The campaign must contain `native/package-degree-4` and the shared
`native/input-turbo-{0,1,2}.json` files. Then:

```bash
python research/h3-scaling/transport/prepare.py \
  --campaign="$CAMPAIGN" \
  --base-package="$CAMPAIGN/native/package-degree-4"
```

For each generated package, run `cozy package lock --json` from that directory,
then `cozy run <package>/fl2va_turbo --describe --json`. Use the configured Hub.
Compare non-root lock entries with the base capture; do not upgrade dependencies
between transport modes. Rent and manage the worker with the ordinary Cozy CLI.

## Clean run sequence

1. Verify four H100 GPU UUIDs, topology, exact software versions, and an idle
   worker. Start passive telemetry and retain its process identities.
2. Run peer warmup, verify all four control receipts and actual NCCL routes, then
   run the two timed prompts. Background downloads and hashing must be quiet.
3. Refresh the worker to the **same** software versions through the ordinary CLI.
   Verify that old executor processes and CUDA contexts are gone.
4. Repeat warmup, route verification, and the timed pair for host mode.

The runner deliberately stops after warmup and after each mode. It never rents,
resets workers, changes software, or resubmits uncertain accepted requests.
Use a unique output directory for each study.

```bash
python research/h3-scaling/transport/run_controls.py \
  --campaign="$CAMPAIGN" --host-dir="$HOST_DIR" --rental="$RENTAL" \
  --mode=peer --phase=warmup

python research/h3-scaling/transport/collect_routes.py \
  --host-dir="$HOST_DIR" --mode=peer

python research/h3-scaling/transport/run_controls.py \
  --campaign="$CAMPAIGN" --host-dir="$HOST_DIR" --rental="$RENTAL" \
  --mode=peer --phase=timed
```

Repeat with `--mode=host` only after the lifecycle boundary. After the timed pair,
collect the final routes with `--case=timed-2`. The host directory supplies the
operator's verified `ssh-command.json` argument array; credentials are not part
of this recipe. The collector reads the actual leader's captured stdout/stderr
file descriptors, rather than guessing a path from its command line.

## What the switch controls

The package's module `warmup()` runs on every rank before communicator creation.
The existing sealed `NCCL_NVLS_ENABLE=0` and `NCCL_P2P_LEVEL=NVL` stay intact.
The only difference between modes is `NCCL_P2P_DISABLE=0` versus `1`. Both use
shared-memory transport when needed, Socket-only network fallback, disabled
GDR/C2C/MNNVL, and the same initialization/transport logging.

A successful environment assignment is **not** transport proof. The parser
requires a completed request, four configuration receipts, the expected NCCL
version, and all 12 directed physical GPU pairs using P2P or SHM as requested.
It maps NCCL's bracketed `nvmlDev` fields to physical GPU identities; communicator
rank labels alone are insufficient because VAE delivery creates two-rank groups.
`SHM/direct` means host shared memory, not direct GPU peer access. A network
fallback is rejected for this study rather than relabeled as SHM.

Native settings affect all NCCL traffic, including metadata and VAE results.
Comfy MultiStream's switch controls only its per-block DiT exchanges. Compare
paired whole-run changes within each implementation; denoising is the closest
shared scope between them.

## Passive link measurements

`sample-links.py` only reads `nvidia-smi nvlink -gt d` and `-gt r`; it never resets
or configures counters. Probe support first and retain the active-link inventory,
GPU UUIDs, exit statuses, timestamps, and raw output. The analyzer expects a
sampler receipt containing `gpu_uuids` and `counter_feasibility` entries named
`status`, `data`, and `raw`, each with `stdout` and `exit_code`.

```bash
python sample-links.py --out="$REMOTE_LOG_DIR" --kinds d r
python analyze_links.py --samples=nvlink-window.jsonl --show=show-final.json \
  --inventory=monitors-start.json --out=nvlink-analysis.json
```

Validate every intermediate sample against the complete active-link inventory
and reject counter decreases or failures. Keep inner and bracketing phase
windows explicit: inner windows omit unsampled edges; outer windows can include
conditioning or VAE traffic. Do not interpolate an exact byte count at a
nominal two-second cadence.

Count selected GPU-port TX once; report RX separately. These are hardware-port
bytes, not unique logical tensor bytes. Convert KiB using 1024, then use decimal
GB/s. Payload/raw queries are staggered, so their subtraction is not exact
protocol overhead. Query-window rate bounds assume no undisclosed NVML cache
lag; `nvidia-smi` does not expose that timestamp. Whole-denoise rates include
compute and are not peak transfer bandwidth. PCIe sampled rates likewise do not
provide exact transfer totals.

Keep CUDA-event collective diagnostics separate from clean timings. Their spans
include packing, metadata synchronization, waits, and possible overlap—not pure
NCCL or global critical-path time. Compare decoded RGB and PCM when MP4 hashes
differ; container mismatches alone do not establish a quality difference.

## CPU checks and sources

```bash
python -m unittest discover -s research/h3-scaling/transport -v
```

- [NCCL environment controls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Pinned NCCL transport selection](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport.cc)
- [Pinned SHM implementation](https://github.com/NVIDIA/nccl/blob/v2.30.7-1/src/transport/shm.cc)
- [NVIDIA counter interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
