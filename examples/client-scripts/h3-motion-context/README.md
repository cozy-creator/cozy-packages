# Matched motion-context comparison

This ordinary client script generates character/scene references once and a ten-second
predecessor once. It then renders three ten-second continuations from that exact same
predecessor, references, prompt and seed, selecting 22, 39 and 56 context frames.
The output is three paired 20-second MP4s. Private context is never a final output.
Logs record sample/delivery clocks, common predecessor/context digests and observed times.
It does not compute a perceptual score or establish seamlessness.

The private schema-2 context contains three independently encoded native audio/video
windows (22, 39 and 56 frames) from the actual delivered predecessor tail. The chosen
window is selected before continuation; a shorter window is never sliced from the
56-frame encoding. The model owns VAE residency while exporting the context, whose
file contract is bounded to 64 MiB.

Prepare an owned qualification directory from the exact candidate serving wheel:

```sh
uv run --no-project python prepare.py /path/to/minimax_h3-1.18.0-py3-none-any.whl /owned/comparison/candidate
cp compare.py /owned/comparison/compare.py
cozy run /owned/comparison/compare.py --rental=your-rental --await
```

The preparer keeps the `minimax-h3` distribution identity and original dependencies so
renderer source inventory and native compatibility checks retain their normal behavior.
It adds an invocable comparison composer to the same App; the real serving children
remain independently captured managed calls. The script imports Runtime's generated
`h3_motion_context.compare` caller. It never loads a model directly.

For an unpublished Runtime/TensorFS cohort, add exact wheel source overrides to both
the owned candidate's pyproject and the script's PEP 723 root metadata. Dependency
source overrides do not implicitly propagate from an editable dependency. Keep those
machine-specific paths and locks outside the distributed repository. Preserve the
explicit Qwen package index in that private capture. Normal published qualification
uses portable package requirements.

The same private candidate retains `long_form` for the separate 60-second composition
qualifications: six ten-second segments, initially selecting 22 context frames, for
both the cinematic fantasy scene and the four-character train action scene. These
continuous outputs remain unqualified until the complete real CLI runs succeed.
It must not be published as the serving release: the extra comparison
entrypoint belongs only to this experiment.
