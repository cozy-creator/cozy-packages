Owner: /root/h3_cozy_astra
Purpose: compile explicit long-form dialogue into official H3 markup with stable speaker IDs and verbatim words.
Branch: fix/h3-structured-dialogue-20260926
Base: a702697 (origin/master fetched2026-09-26; includes existing-reference/style/audio API PR278)

Scope: deterministic per-shot dialogue plus explicit placement markers; no prose-quote guessing, inference, rentals, tests, CI, lint or vet. Only source review and model-free CLI contract/rendered-example checks. Running request1128, its local candidate, installed packages and global SDK remain untouched. Public publication remains held.
