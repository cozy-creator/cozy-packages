# SDXL model assets

SDXL's tokenizer vocabularies and settings are part of the selected model
checkpoint, not the package release. The package consumes the Runtime model
asset view during `SdxlModel.load` and never reads a source-tree path or the
Hugging Face network.

The selected CozyTensors asset map must contain these names:

```text
tokenizer/vocab.json
tokenizer/merges.txt
tokenizer/tokenizer_config.json
tokenizer_2/vocab.json
tokenizer_2/merges.txt
tokenizer_2/tokenizer_config.json
```

`special_tokens_map.json` is not read by SDXL and is intentionally not part of
the required runtime closure. Runtime materializes a bounded, read-only view
only while constructing each `CLIPTokenizer`; the parsed tokenizer then remains
in memory for warmup and requests.

The current published checkpoint predates this contract and has an empty asset
map. It must receive a metadata-only replacement snapshot before this package
release can be served. The replacement must reuse every existing tensor object
reference and change only the CozyTensors header and manifest identity.
