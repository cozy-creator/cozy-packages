# Ordinary Cozy interleaved memory benchmark

Prepared CPU-only by `audit_rollout_completion`. Do not execute until root grants the exclusive GPU window and installs the agreed CLI/Runtime cohort. No default-home package, machine, daemon, budget or GPU changes were made during preparation.

`input-manifest.json` freezes the shared **final-landscape-v1** specification, canonical model checkpoints, exact payloads/seeds, source hashes and private package locks. All A/B/Anima prompts are the same benign landscape; the explicit negative includes nudity. Prior generic-prompt cells are diagnostics only and are not scored alongside this specification.

- Alternating: A, Anima, A, Anima; seeds1005–1008.
- Reuse: A, A, B, B, A, Anima, Anima; seeds1005–1011.
- A=paul/sdxl@1.0.0; B=paul/wai-illustrious@17.0.0; Anima=paul/anima@1.0.0. Every full receipt must match the frozen checkpoint digest.
- 1024×1024. SDXL20steps/CFG7/HiDiffusionoff. Anima30steps/CFG4.5/full CFGinterval0..1/FBC0/empty quality prefix.
- Private package names avoid replacing earlier local installs. Author modules are byte-identical to the source package worktree. Only private distribution name, Python3.12 and dependency sources/locks change.
- The private R5 wheel is0.18.88+dev.h2756… (SHAa56720…). **Public0.18.88 belongs to the separate boundary-only release; public memory release target is>=0.18.89.** Original package minimums remain untouched.
- Torch2.14.0+cu130 and torchvision0.29.0+cu130 (Anima) match Comfy's recorded wheels. TensorFS0.3.78 and the exact R5 Runtime are local wheel sources. These private paths are never publication metadata.

## Execution, only after handoff

Use a Python environment containing psutil, nvidia-ml-py, Pillow and PyYAML (the Comfy benchmark environment already has these). The driver invokes inference **only through ordinary `cozy run` in the default home**. It never runs model Python directly, buys rentals, cancels runs, stops daemons/machines, removes packages, or edits configuration. On error it retains owned IDs and submission intents; accepted work continues. Never blindly rerun a failed output directory.

Root first installs the agreed default CLI and R5 machine pair at an idle boundary. `cozy machine show` must report stopped before every fresh-engine cell. `gpu.budget` must be off/absent. Cold worker package environment preparation remains inside primary timing. Optional client package registration is separately reported; inclusive timing is also retained.

Example (paths must be absolute):

```sh
systemd-run --user --scope --slice=app.slice --collect --quiet \
  --unit=cozy-memory-benchmark-r5-landscape-uncapped-alternating \
  /path/to/comfy/.venv/bin/python /path/to/benchmarks/cozy_memory_interleaved/run.py \
  --suite alternating --budget-label uncapped \
  --out /home/fidika/cozy_v2/outputs/comfy-cozy-memory-20260929/cozy/final-landscape-uncapped-alternating \
  --peer-environment /path/to/matched-comfy-server-environment.json \
  --register-packages --execute-gpu --ownership-handoff 'root handoff evidence'
```

Registration is needed once for the owned package clones; later cells may omit it. It does not remove older installs. No `--rental`, `--rental-only`, or `--rent-new` flag is ever sent. Default current Creator execution is local-only, and every observed receipt must name local.

For3GiB/1GiB, root runs the **same Comfy ballast.py helper** before either fresh engine, with its explicit GPU gate and fresh receipt/release-file paths. Pass `--ballast-receipt`. This is physical free-device-memory pressure, not a Cozy configured grant quota. Driver verifies the ballast's live PID and calibrated target, records its cgroup and pre-engine free delta, and refuses an unexpected compute process. The helper's owner releases it after accepted work settles; this driver never kills it.

The driver submits one run, waits for its recorded open GPU lease, and only then submits its successor. Comfy enqueues its entire FIFO sequence up front; this Cozy-specific one-successor policy is explicitly different. At most one pending successor permits prefetch without allowing A/Anima/A/Anima to coalesce into A/A/Anima/Anima. Final actual lease ordering is checked, not inferred from submission order. Seeds are unique within a suite and there is no output memoization/retry shortcut.

## Timing and receipts

Primary common comparison is **cold engine startup plus the entire sequence**. Cozy starts the empty machine implicitly in its first ordinary `cozy run`; measure that invocation to the last export file mtime. Comfy starts its empty server first, so combine its separately measured startup_ms with ready-server first-submit-to-SaveImage-event time. Preserve the small Comfy ready-to-WebSocket-to-submit gap as an explicit gap if not included. There is no ordinary empty-machine start command; do not add a dummy inference or invent a Cozy startup subtraction. Runtime source/environment installation, preparation, model loading, execution, WebP encoding and ordinary export stay inside primary wall. Client package registration is separately timed and an inclusive registration-to-output metric is also emitted. Cozy also records monotonic verified-export observation end, verification/observer delay, all CLI durations and export filesystem timestamps. The verification endpoint includes RGB hashing and polling; it is never aliased to Comfy saved-event time. `ready_service_queue_ms` is unavailable on Cozy; existing setup/GPU stages remain diagnostic rather than an invented readiness timestamp. No measured source setup time is subtracted from the cold total.

Every output is checked for receipt size/SHA256 and1024² decode; RGB SHA is recorded. Every completed run retains full CLI JSON with model/Runtime identity, closed ledger, setup stages, execution windows, tier/movement metrics, GPU PID/reuse evidence and result/decode mode. Low-rate CPU/RSS/IO and NVML samples cover the run. These are process counters and Runtime phase measurements, not an intrusive CPU stack/kernel profiler.

Cozy outputs WebP; Comfy SaveImage outputs PNG. Encoding/export cost is included, and same seeds/sampler families are not a claim of bitwise image equivalence across frameworks. Selected VAE mode is recorded; authored package policy is not modified for a better score.

## Cgroup matching

Old Comfy cells ran inside shared `agents.slice`: MemoryMax40GiB, MemoryHigh32GiB, CPU quota20cores, TasksMax4096. Default Cozy's independent systemd user unit has a different ancestor. Those old cells are **not a matched RAM/CPU comparison**.

Root approved fresh Comfy runs via `systemd-run --user --scope --slice=app.slice`, outside `agents.slice`, without editing any existing unit. Run the Cozy driver in its own equivalent scope too; the ordinary machine still uses its own unit. `environment.py:process_environment(pid)` records PID/startticks, affinity and effective ancestor limits. Save that function's Comfy server result as `--peer-environment`. Cozy records its actual machine PID environment after startup and marks results scored only if effective limits and affinity match and neither is inside agents.slice. Record ballast cgroup separately. No cgroup settings are changed by these scripts.

The scope is collected after its owned driver/children exit. Default machine shutdown and subsequent benchmark ownership are root's decisions after the ledger/owned process census, never automatic cleanup of historical paused/export rows.

## Static verification

`prepare.py` parses author input fields through AST, validates every payload, checks immutable checkpoint proof references, verifies unchanged author files, freezes exact locks and matches shared specSHA. `run.py --plan-only` starts nothing. Both uv locks passed `--check`; scripts compile and basic payload/seed/lease/cgroup controls were exercised without CLI or GPU invocation. No inference result is claimed before the authorized run.

## Current R6 handoff

R5 input/package/code snapshots remain intact and are not to be executed. The generic driver now **requires** `--manifest`; select `cohorts/r6/input-manifest.json`. R6 is the private0.18.89+dev.h481120… artifact, wheelSHA7180737f55a667235b0ae4b64a9b19094cf94186e079ebbc119e6c0e110abf69 and bundled agentSHA6fbd12a494045e6d6da380b9e3d9d3976662d8c01efdf1649b2a72bae3f8ef1d. `prepare_cohort.py` makes fresh cohort directories and refuses overwrites. It has no install/daemon/GPU actions.

The prepared CLI for root's future installation is `/home/fidika/cozy_v2/outputs/comfy-cozy-memory-20260929/cozy-cli-71b84a10/cozy`, SHA1ed0aaad22fed19d344720410b17b66778ceab7c5649449df8c2171d4c8410d8. No installation has been performed here. The matching Comfy peer environment receipt is `comfy/final-landscape/comfy-peer-environment.json` under the shared output folder.

## Filesystem-cache fairness

Comfy verifies/hashes its raw weight files before engine startup. The Cozy driver now calls `prewarm.py` before each cell, after optional controller registration and before initial GPU measurement/first run. It uses public `tensorfs.Store.open`, `manifest`, `acquire_cozytensors` and `ReadLease.read_into`, verifies every native object SHA256 with one8MiB client buffer, and closes each read lease. It never imports Torch, starts a machine, downloads/materializes weights, adds retention references or flushes caches. The three fixed checkpoints are supported; each cell reads the same required model set as Comfy (A/Anima for alternating; A/Anima/B for reuse, matching source-group traversal). Headers/assets in the native closure are also read. Repeated CAS objects may be shared; byte counts are reported.

Prewarm duration, CPU time, read blocks and peakRSS are separate from the cold-engine sequence score. This creates a **verified warm intent**, not guaranteed page residency: buffers are bounded, pages are not pinned, and the shared64GB host can evict either stack's different backing files. Do not describe this as matched cold storage or guarantee all weights remain in RAM. Actual process IO telemetry remains available. No bulk default-store prewarm is performed during CPU preparation while Comfy owns its cache.

The isolated reader environment is `outputs/comfy-cozy-memory-20260929/cozy/prewarm-venv`, containing only the exact TensorFS0.3.78 native wheel. A tiny isolated real-TensorFS checkpoint test passed with an8KiB buffer and no default-store/GPU access. CPU mock driver tests exercised both complete4/7-job sequences without invoking a real CLI, sampler or prewarmer.
