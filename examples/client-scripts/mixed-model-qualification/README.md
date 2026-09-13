# Real rental mixed-model qualification

This is an unpublished test package and two one-off client scripts, not a published
model tools API. Use the ordinary `cozy` CLI and default `~/.cozy`. Root supplies
an approved, compatible rental; do not automatically buy a machine or touch another
owner's rental. Code and two tiny 2x2 checkpoint matrices are the only task inputs.

The package requires actual CUDA execution but no native custom kernel or large
model. Its native output grant is 4KiB; each checkpoint has 32 bytes of tensor payload
plus a native header and inline JSON config. All imports are at module scope.
The independently captured package lives in `library/`; caller scripts stay
outside that project so editing their scenarios does not invalidate its memo key.

## Prepare the published default

From this directory, with `RENTAL` set to root's approved rental name:

```sh
cozy run ./seed_base.py --rental="$RENTAL" \
  --publish-to=paul/cozy-mixed-input-proof --org=paul --await --json
cozy model publish paul/cozy-mixed-input-proof \
  --release=0.0.0-rental-audit.20260913 --lane="f32=$CHECKPOINT" --json
```

Set `CHECKPOINT` from the actual returned checkpoint manifest ID, not a receipt or
publication ID. Read the release back with `cozy model info` before continuing.
The seed script imports only a plain helper; it must not attempt to bind or download
the inference function's default model before that model has been published.

## Mixed input and reuse

```sh
cozy run ./compare.py --rental="$RENTAL" --await --json
cozy run ./compare.py --rental="$RENTAL" --await --json
```

Plain script `main` accepts context and injected services rather than request
arguments. Set `NOTE` and `HOLD_FOR_CANCEL` at the top of this one-off script when
selecting the cancellation scenario, then submit with the same command without
`--await`. Return `HOLD_FOR_CANCEL` to `False` for the post-cancel comparison.
The small comparison report is returned as JSON text under `result.value`.
Only invocable exports are imported from the generated client library; report
formatting stays in this one-off script.

Each comparison creates two separate inference requests with identical seed 24680
and two steps. Both slots execute different real weights: a newly produced retained
scale 2 checkpoint plus the published default scale 3 checkpoint. The expected ratio
is 2.25. Compare their returned checkpoint/request IDs and numerical/RNG results.
The repeated parent must reuse the native producer, as confirmed by ordinary
`cozy run list --json --full` and `cozy run watch` records; equal bytes alone do not
prove reuse. A caller-only comment edit can additionally qualify edited capture.

The hold run acquires the candidate and starts one held serving child without
repeating the already-proven numerical comparisons. Wait for its `awaiting-cancel` stage, then use
`cozy run cancel <parent-run> --json`. The parent and active child should settle as
cancelled, their active input custody should release, and the original completed
producer must remain reusable. Run `cozy run ./compare.py
--rental="$RENTAL" --await --json` afterwards and compare its producer ID with
the original; a pre-cancel reuse alone is not a post-cancel custody proof. Stop/restart `cozy run watch <run>` to test client
reconnection without cancelling work. Cache-miss recovery is automatic through
normal execution, but perform eviction only through an explicitly approved ordinary
CLI/storage operation; do not use direct RPC or manipulate worker journals.

## Success and cleanup

Keep CLI JSON results/events with the actual cohort, rental and run identities.
Require a real catalog checkpoint/release readback, four successful CUDA calls,
correct distinct-weight scaling, same-seed equality, independent call identities,
producer reuse and accepted-call cancellation. Report reconnect/cache-miss coverage
separately if it cannot be exercised by an ordinary authorized path.

No daemon restart, SDK/binary installation, worker update, rental creation/end or
catalog deletion is automated by these scripts. Root owns those operations.
Leave the clearly named test release as evidence unless root asks to remove it;
never revoke the only retained copy of a candidate before verification completes.
