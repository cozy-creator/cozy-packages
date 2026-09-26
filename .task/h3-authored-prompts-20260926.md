# H3 official reference prompt layout and per-shot audio

- Owner: /root
- Tracker: https://github.com/cozy-creator/tracker/issues/240
- Branch: fix/h3-authored-prompts-20260926
- Base: a4617d7f71c27b0e23c28d37a75e53a35c7166c8 (fresh origin/master)
- Purpose: follow the official six-section reference guide using an explicit segments list with authored summary, detailed_description, overall_soundscape and non_diegetic_music. Global references, retention and style stay shared; camera-shot numbers are local to each invocation. The user's final segment contract supersedes the earlier minimal layout and optional audio-override draft.
- Delivery: shared source, normal paul/minimax-h3 local-Hub update, migrated input JSON, and one file containing all nine exact prompt blocks.
- Validation: source review and model-free prompt rendering; no CI, tests, GPU or rentals. Local-Hub release/install only, using verified available public Runtime 0.18.32, TensorFS 0.3.55 and Eval 0.7.4 dependencies; no production-Hub changes.
