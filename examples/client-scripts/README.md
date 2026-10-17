# Private client scripts

Run a Python file directly with `cozy run ./path/to/script.py --on any`.
The client captures the script and its editable dependencies, then executes the
ordinary `main()` function on the private worker. These files are not published
packages and have no `package.toml`, App registry, or request/result classes.
Current private capture and preparation require Runtime 0.16.10 or newer on
both the local client and private worker. The script captures its dependencies.

Edit the script and run it again. Each invocation starts at `main()`, while the
library operations look up compatible completed results and validate that their
bytes still exist. A caller-only edit does not invalidate their keys. Changed
operation inputs, parameters, plan or relevant implementation produce a new key.
The native conversion and quantization writers can also adopt compatible
partial work; this is explicit library support, not Python line replay. A Runtime
upgrade can conservatively invalidate more work than a caller edit.

The same Runtime/TensorFS code is used locally and on a private worker. Each Store's
`.cozy-workspace/journal.sqlite3` records operation identities, results, progress and
retention; TensorFS stores the artifact bytes. Reuse requires that Store and its
retained bytes. There is no need for a previous run ID or Tensorhub publication.
Inference runs fresh; checkpoint upload and release creation are explicit effects.

Scripts bind models with ordinary `model.<slot>=<ref>` terms. Optional PEP 723
`[tool.cozy.models]` entries can set these defaults; explicit CLI terms take precedence.
Imports and model classes come from the captured library environment. Script-local classes
and annotation expressions with function calls are not capability declarations.

Completed memoized operations and independently retained partial work can be reused
by a new script on the same worker. A script itself starts from `main()` each time;
plain Python lines are not replayed or skipped by a workflow engine.
