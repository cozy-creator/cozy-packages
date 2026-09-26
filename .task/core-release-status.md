# Completed local core release cut

Normal global `cozy package publish --tensorhub=http://127.0.0.1:8819 --wheel=... --json --full` published:

| Reference | Project wheel SHA256 |
| --- | --- |
| paul/qwen-image-2@0.2.0 | e7ad3e3eddfdc276546937e7ea9bc9b6cea32c83ec62c6bb33edb240650f31ac |
| paul/minimax-h3@1.18.1 | b14fd7672b36c1fa6c9f131b4ff62d6e5705524ddfdcd958fe13910f9177a2a8 |
| paul/sdxl@2.3.17 | 0014c770131f662a406c9e7ac547c88de077f390e75e5b7f1b80a8057ae36079 |
| paul/minimax-h3-tools@2.12.10 | 728620d095ae9525b42c48f0b1624df6cac51138c72c24bc68c513d3146e3ab7 |

All four versions have CLI authored-binding readbacks and matching standard-index hashes from `uv pip compile --no-deps --generate-hashes`. H3's local publication lock selects Runtime0.18.32, Eval0.7.4, Qwen0.2.0 and TensorFS0.3.55. The user's local/minimax-h3 installation4b1e5e1cce171208 and editable source path are unchanged. No production Hub or rental was modified.

Qwen/H3 strict mypy passes against Runtime's corrected counted-default annotations (Runtime656). Five changed SDXL/H3tools memo files also pass. Exact Qwen/H3 candidate wheel dynamic interfaces equal the committed static bytes in a bounded cached CPU source-contract check; this is not numerical or GPU qualification.

Normal Qwen publication exposed Hub's scalar-only default gate and unretryable failed-finalization state. Hub770/771 fixed both; the unchanged publication intent then completed using its already staged objects. No database reset or package-version workaround was used.

Evidence: /home/fidika/cozy_v2/outputs/core-local-release-20260926/release-summary.json, published-hashes.txt and per-package publish/readback JSON.

## Separate production gate

Qwen/SDXL/H3tools portable locks are committed. H3's requested local release used an owned source copy with an explicit local Hub index and lock. The repository retains its canonical production index; its H3 lock cannot be refreshed to Qwen0.2.0 until a production Qwen publication is separately authorized. The local release is complete; production publication and GPU qualification are separate gates.
