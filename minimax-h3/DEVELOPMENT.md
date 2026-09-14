# Runtime model-library dependency

This workflow imports its standard and Turbo models from `cozy_runtime.models.minimax_h3`; it no longer owns a second inference implementation. The `h3.H3Model`, `h3.H3TurboBase` and `h3.H3TurboLoRA` names remain explicit imports for existing diagnostic/custom workflow consumers.

This source requires the Runtime built-in model-library change, paired Runtime PR #508. The older published `cozy-runtime==0.18.2` wheel does not contain that API even though the development wheel currently uses that version. Do not publish or activate this workflow with that older artifact. Use the matching, content-addressed development Runtime wheel for qualification. The normal version dependency will be tightened when a real release contains the API; CI substitution is not evidence that the older release supports it.

The local paired-wheel proof includes installed workflow imports, static-interface equality, exact model/component declarations, and retained-prefix source provenance. Prefix provenance now hashes Runtime's actual H3 code/assets in addition to workflow bytes and dependency versions, so two development artifacts with the same version are not considered equivalent.

The package's checked-in schedule JSON remains solely because static `data_values` describes the request's step enum without executing code. `scripts/h3-conform.py` checks byte equality against Runtime's canonical resources. Tokenizer/processor resources and all executable H3 inference helpers live only in Runtime.

GitHub CI currently installs the published Runtime. Cross-repository source qualification requires an existing read credential for the private Runtime repository or an explicitly authorized artifact delivery path. No credential or green-check bypass is introduced here; the paired PR remains draft until that delivery dependency is satisfied.
