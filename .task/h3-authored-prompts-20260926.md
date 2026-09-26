# H3 official reference prompt layout and per-shot audio

- Owner: /root
- Tracker: https://github.com/cozy-creator/tracker/issues/240
- Branch: fix/h3-authored-prompts-20260926
- Base: a4617d7f71c27b0e23c28d37a75e53a35c7166c8 (fresh origin/master)
- Purpose: follow the official six-section reference guide, with the authored shot directly after [Shot 1], proper dialogue tags, subject bindings and per-shot audio overrides. The user's later guide request supersedes the earlier minimal layout.
- Delivery: shared source, normal paul/minimax-h3 local-Hub update, migrated input JSON, and one file containing all nine exact prompt blocks.
- Validation: source review and model-free prompt rendering; no CI, tests, GPU, rental or public publication.
