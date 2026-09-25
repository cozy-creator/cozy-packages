# SDXL tokenizer assets

SDXL reads optional tokenizer overrides from the selected CozyTensors checkpoint.
For each tokenizer, the vocabulary and merges form one pair:

```text
tokenizer/vocab.json
tokenizer/merges.txt
tokenizer_2/vocab.json
tokenizer_2/merges.txt
```

When neither member of a pair exists, the package reads its bundled standard CLIP
vocabulary and merges. Both SDXL encoders share those bytes, so the package stores
one copy as `sdxl/clip_vocab.json` and `sdxl/clip_merges.txt`. They are restored
unchanged from the package's original tokenizer data at `d9ef626^` (the two encoder
copies had identical Git blob IDs). Wheels and source distributions include them.

A complete checkpoint pair takes precedence. A partial pair is an invalid
override: the missing asset still raises Runtime's `model_asset_missing` error.
Invalid bytes and other Runtime asset errors also propagate. The fallback never
combines a checkpoint vocabulary with package merges or the reverse.

The first CLIP tokenizer pads with `<|endoftext|>`; the second pads with `!`.
Both use replacement errors and a 77-token maximum. Unused tokenizer settings
sidecars do not change those model-family constants.

Both sources feed the same in-memory tokenizer constructor. Loading reads files
or verified model bytes without network access or filesystem writes, including
inside Runtime's admission sandbox. The constructed tokenizers remain attached
to the loaded model and are reused by warmup and every request.

`scripts/tokenizer-assets-proof.py` exercises the real Runtime read-only sandbox,
fallback and override behavior, partial and corrupt overrides, and Anima's
bundled tokenizers. Pass built wheels with `--wheel` to verify their package data.
