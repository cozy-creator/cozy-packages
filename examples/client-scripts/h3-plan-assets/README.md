# H3 timestep asset migration

Generate the historical base and Turbo plan assets with `scripts/h3_plans.py --output DIR`
(base: `30 40 50`; Turbo: `--turbo 8`). Prepare a JSON request containing `source_repo`,
`source_revision` (the exact `sha256:` tree), `endpoint`, `plan_dir`, a new `work_dir`, and
`turbo` (boolean), then run locally:

```sh
cozy run ./examples/client-scripts/h3-plan-assets/prepare.py --asset request=./request.json --on local
```

The script reads only variant headers and repository files. It creates `source/`, `proposed/`
and `receipt.json` under the requested work directory. Native readback and structural equality
checks verify that only plan references were added; tensor references, configs, other assets,
README and LICENSE remain unchanged. It refuses legacy generating-projection bindings.

Preparation does not upload or publish. Upload `proposed/` through a CLI supporting CozyTensors
folder sources, verify the private commit against the receipt, then publish the replacement
model release before releasing consumers. Preserve the old model releases.
