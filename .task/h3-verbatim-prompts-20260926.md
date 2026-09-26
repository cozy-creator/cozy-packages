# Verbatim H3 authored text

- Owner: /root
- Tracker: https://github.com/cozy-creator/tracker/issues/240
- Branch: fix/h3-verbatim-prompts-20260926
- Base: 3ad8067f479c52470a6438bda1d185de3226af13
- Purpose: user explicitly requests natural H3 text, with `<Reina>` reaching the model unchanged. Remove custom placeholders, marker expansion, tag rewriting and quote cleanup.
- Contract: shared authored subject_definitions, retention_analysis and style; four authored strings per segment. Only section headings and the shared style are assembled.
- Release: prior 1.18.2 wheel is held, never published/installed. New reviewed source replaces it before normal local-Hub publication/install. No production Hub, CI/tests, GPU work or rentals.
