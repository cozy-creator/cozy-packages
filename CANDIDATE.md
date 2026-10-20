# Varena slim candidate checkout

This branch is an isolated linux x86_64 proof checkout, not a package release. Production locks, package versions and PR518 remain unchanged. Every package lock here uses the real Runtime0.30.0 and TensorFS0.13.0 prerelease wheels listed in `scripts/varena-slim-candidate-requirements.txt`. PyAV and the H3/SDXL PEFT dependency are explicit because these Runtime wheels do not provide the former media/adapter extras.

Use this checkout with the matching candidate `cozy` client and candidate worker. From its root:

```sh
cozy run ./sdxl/generate --on rent-19f198a5 prompt='A red fox' seed=42 model.model='<private-sdxl-ref>/int8'
cozy run ./minimax-h3/ref2va_turbo --on rent-19f198a5 --input <request.json> model.local/minimax-h3/generate_ref2va_turbo.models.base_model='<private-h3-ref>/int8' model.local/minimax-h3/generate_ref2va_turbo.models.turbo_lora='<private-turbo-ref>/pdd8'
```

`ref2va_turbo` is a CPU wrapper with no model slots. Its internal `generate_ref2va_turbo` owns `base_model` and `turbo_lora`. `long_form` uses `motion_segment` and `encode_text`; `long_form_cuts` uses `cut_segment`. Set those internal slots explicitly. `candidate-model-slots.json` records their exact private tree refs and variants, plus the Qwen, Klein and character-describer slots. The publication owner verified all23 variants through fresh-cache native digest opens; all nine repos remain private and unreleased, and public source documents/ETags are unchanged. Klein's drafter uses bf16.

H3's Qwen, character-describer, BiRefNet, SeedVR2 and table-producer dependencies are sibling path sources. Creator carries their source directories with the local project. Their identities are `local/<distribution>`, rather than the published `fidika/<distribution>` identities. Thus Qwen reference-image and character-description child overrides use, for example, `model.local/qwen-image-2/generate_image.models.model=.../int8` and `model.local/character-describer/write_texts.models.model=.../int8`. Internal H3 self-calls stay `local/minimax-h3` and use this exact source. No published H3 package participates.

Hosted qualification uses the unchanged `task conform -- <package>` checks for all nine package environments, including native imports, current interfaces, model proofs and installed-wheel imports. This branch's static job installs the same SDK URLs. Do not run the repository's publication task on this candidate branch.
