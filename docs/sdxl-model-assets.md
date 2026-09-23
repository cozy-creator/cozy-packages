# SDXL model assets

SDXL's tokenizer vocabularies and settings are part of the selected model
checkpoint, not the package release. The package consumes the Runtime model
asset view during `SdxlModel.load` and never reads a source-tree path or the
Hugging Face network.

The selected CozyTensors asset map must contain these names:

```text
tokenizer/vocab.json
tokenizer/merges.txt
tokenizer_2/vocab.json
tokenizer_2/merges.txt
```

Tokenizer behavior is a small reviewed model-family constant: the first CLIP
tokenizer uses an `<|endoftext|>` pad token and the second uses `!`; both use
replacement errors and a 77-token maximum. The unused `tokenizer_config.json`
and `special_tokens_map.json` sidecars are not model assets. Runtime materializes
only the bounded vocabulary/merge files while constructing each tokenizer; the
parsed tokenizers then remain in memory for warmup and requests.

The current published checkpoint predates this contract and has an empty asset
map. It must receive a metadata-only replacement snapshot before this package
release can be served. The replacement must reuse every existing tensor object
reference and change only the CozyTensors header and manifest identity.
