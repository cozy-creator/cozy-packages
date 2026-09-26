# H3 prompt quotation cleanup

- Owner: /root
- Purpose: remove non-dialogue quotation delimiters and shot-description metainstructions from compiled long-form H3 prompts after reported instruction speech.
- Branch: fix/h3-prompt-quoting-20260926
- Base: a1bc18d2e14e8b953e6ce1fad3c16a701b92ddf1 (fresh origin/master)
- Tracker: https://github.com/cozy-creator/tracker/issues/238
- PR: https://github.com/cozy-creator/packages/pull/284
- Scope: package source, examples and documentation; independent source review and CPU prompt rendering only.
- Constraints: no CI/test suite, GPU, rentals, active candidate edits, installation or publication. The public release remains held.
- Workspace: retained owned worktree; legacy policy is absent, audit recovered at ~/cozy/.cache-graveyard/tracker-shared-scripts/workspace-git-audit.py. No cleanup.
