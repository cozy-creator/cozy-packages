# H3 generated retention removal

- Owner: /root
- Tracker: https://github.com/cozy-creator/tracker/issues/240
- Branch: fix/h3-remove-retention-20260926
- Base: c095d80b69467c67649441a4386c48cb5733e95a (fresh origin/master)
- Purpose: remove generated retention_analysis from actual long-form model prompts, at the user's explicit request.
- Delivery: shared source plus narrowly backported change to the active matching local candidate; refresh through ordinary CLI and render the user's nine-shot JSON into one Markdown file.
- Validation: source review and CPU prompt rendering only; no tests, CI, GPU work, rentals or publication.
- Workspace: primary checkout has unrelated untracked work and is preserved. Legacy workspace policy is absent; recovered read-only audit is available in ~/cozy/.cache-graveyard/tracker-shared-scripts.
