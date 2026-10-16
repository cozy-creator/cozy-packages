# First-party package releases

Publish a merged package, and everything that locks it, to both Hubs:

```sh
task publish -- qwen-image-2
```

It fast-forwards the checkout to `origin/master`, runs the package's CPU conformance
(`task conform -- <package>`), and publishes a `git archive` of it to each of `HUBS`.
Then, in dependency order, each package whose `uv.lock` holds a republished package is
relocked to it (`cozy package lock --tensorhub tensorhub --upgrade-package <it>`, plus
`cozy-runtime` or `tensorfs` when its floor is above the dependent's lock),
patch-bumped, conformed, pushed to master and published the same way. A failed
conformance stops the cascade before that package is committed. A dependent whose
master version is unpublished stops it too: its merged changes are not released yet.
Republishing a release is a no-op, so rerun the same command after a fix.

Complete the [dependency qualification checklist](DEPENDENCIES.md) first. Worker images
are OS/CUDA/PyTorch bases named by tag; the CLI updates the machine, Runtime and TensorFS
at runtime, so a package never waits on an image.

## One source, every Hub and account

Package source names no account. A package publishes into the namespace of the
account that runs `cozy package publish` on the command's Tensorhub: `paul` on
the local `127.0.0.1:8819` Hub, `fidika` on tensorhub.com. Default lanes for the
publisher's own models omit the org (`minimax-h3@1.0.0-rc.2/fp8-pruned`); Creator
writes the account into the published interface. Same-account package dependencies
name the `tensorhub` index (`qwen-image-2 = { index = "tensorhub" }`).

Commit locks bound to production. After changing dependencies, run as `fidika`:

```sh
cozy package lock --tensorhub https://tensorhub.com
```

A published release runs exactly the dependency versions in its `uv.lock`, including
same-account callees such as qwen-image-2, so each callee release needs a dependent
release; `task publish` makes it. The exception is `cozy-runtime` and `tensorfs`: every
package runs on its machine's Runtime, so the package only declares the minimum its
`cozy-runtime>=` bound admits.

`scripts/bind-account-index.sh` writes the same index for the CI lock checks. Each
publication rebinds the committed lock to its target: `cozy package publish` to the
local Hub as `paul` relocks a private copy against `paul`'s index there and refuses
(`account_index_lock_drift`) unless every locked version exists unchanged, which is
why `task publish` releases dependencies before dependents on each Hub.

Prefer lower bounds for package dependencies, such as `pydantic-core>=2.46.4`.
If a compatibility ceiling is needed, bound the major version (`>=2.46.4,<3`) or
at most the minor version (`>=2.46.4,<2.47`). Major/minor wildcards and equivalent
compatible-release ranges are allowed. Exact releases (`==2.46.4`, including
short forms such as `==2.46`) and patch ceilings are rejected. Keep the exact
tested versions in `uv.lock`; compatibility declarations must admit patch updates.

When a package starts using a new Runtime API:

1. Release and verify Runtime, including its public wheel artifacts.
2. Set the package's minimum Runtime version to the oldest version that supplies
   the APIs it actually uses, relock, and run package conformance.
3. Publish with `task publish`.
4. Install the published release with the local Creator CLI and verify its input
   and callable interface. Include execution qualification appropriate to the change.

Private development or custom-image packages can still use `cozy package publish`
directly. Their requirements need not fit every first-party worker image. The
worker capability check remains authoritative for new worker APIs, and an existing
private rental may need `cozy rental update <name>` before using such an API.
Ordinary dependency updates belong to the package environment and do not require
updating the worker's global Python environment.

## H3 1.24.3 qualification

H3 1.24.3 requires Runtime 0.18.89 for tracked background execution. Its production
lock changes Runtime 0.18.73 to 0.18.89 and TensorFS 0.3.74 to 0.3.78. Torch 2.14.0,
torchvision 0.29.0, Qwen Image 2 version 0.2.3, and Cozy Eval 0.7.5 remain unchanged.
The one-image character sheet prompt from 1.24.2 and the callable interface remain
unchanged.

Qualification uses the published Runtime 0.18.89 and TensorFS 0.3.78 with the CPU
prompt, long-form, assembly, and progressive-join proof scripts. The long-form
fixture checks complete execution observations for the composer and each child.
The progressive proof checks immutable live fragments, copied video packets,
decoded media, final duration, and seeks between keyframes. Rendering in these
proofs is synthetic; the checks validate workflow, custody, and codec behavior.

Publish the same frozen H3 source to each Hub; each account's dependency index must
resolve the committed versions.
