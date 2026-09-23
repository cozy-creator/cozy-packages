# First-party package releases

Publish first-party packages through the checked release command:

```sh
uv run --no-project python scripts/publish-package.py minimax-h3 \
  --tensorhub ../tensorhub \
  --dotenv ../tensorhub/.env \
  --profile torch2.14.0-cu130-cp312-linux-x86 \
  --check-only
```

Run this from the packages checkout after committing the package source and
lockfile. `--tensorhub` points at the Tensorhub operator checkout that supplies
the active worker-image records. `--dotenv` is a local path to that checkout's
operator environment; keep the file outside version control and never copy its
values into release notes or issue reports. The current CUDA worker profile is
`torch2.14.0-cu130-cp312-linux-x86`.

The preflight builds an immutable snapshot and checks its authored Runtime
requirements against the active normal and private worker images. A missing
image or incompatible Runtime API stops this first-party release preflight. The
check does not prove that the package's full dependency environment installs or
executes. Complete the
[dependency qualification checklist](DEPENDENCIES.md) as well. Baked application
dependencies are a reuse optimization; a version mismatch must be resolved in
the package's isolated environment rather than treated as a global image limit.
After the read-only preflight passes, rerun the same command without
`--check-only` to publish the committed snapshot through the ordinary Creator
CLI. The Tensorhub operator checkout supplies the read-only image check; its
normal configuration flags select the environment being released to.

Prefer lower bounds for package dependencies, such as `pydantic-core>=2.46.4`.
If a compatibility ceiling is needed, bound the major version (`>=2.46.4,<3`) or
at most the minor version (`>=2.46.4,<2.47`). Major/minor wildcards and equivalent
compatible-release ranges are allowed. Exact releases (`==2.46.4`, including
short forms such as `==2.46`) and patch ceilings are rejected. Keep the exact
tested versions in `uv.lock`; compatibility declarations must admit patch updates.

When a package starts using a new Runtime API:

1. Release and verify Runtime, including its public wheel artifacts.
2. Build and qualify the supported worker images with that Runtime, then promote
   their immutable image records. Keep existing rentals and cached work intact.
3. Update the package lockfile and run package conformance. Set its minimum
   Runtime version to the oldest version that supplies the APIs it actually uses.
4. Run the checked publication command above. Repeat `--profile` for every
   supported image profile. Use `--purpose` only when a package explicitly supports
   one purpose; the default checks both.
5. Install the published release with the local Creator CLI and verify its input
   and callable interface. Include execution qualification appropriate to the change.

Private development or custom-image packages can still use `cozy package publish`
directly. Their requirements need not fit every first-party worker image. The
worker capability check remains authoritative for new worker APIs, and an existing
private rental may need `cozy rental update <name>` before using such an API.
Ordinary dependency updates belong to the package environment and do not require
updating the worker's global Python environment.
