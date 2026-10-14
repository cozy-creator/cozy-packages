# Character describer

`caption_images` describes each supplied character image in one sentence: identity, hair,
face, clothing and carried or worn items, never the background, pose or style. Images are
described independently, in request order, with greedy decoding. A caption that names a
setting or pose word anyway is regenerated once with those words pointed out, and any clause
that still names one is dropped. A request takes 1 to 12
PNG, JPEG or WebP images of up to 64 MiB each; each caption carries its image's `index`.
MiniMax-H3 uses it to fill a supplied character's missing description.

```sh
cozy run fidika/character-describer/caption_images --asset images.0=./character.png --await
```

The default model is Qwen3.5-9B (`qwen3.5-9b@0.1.0/bf16`), chosen over Qwen3.5-4B,
JoyCaption Beta One and Qwen3-VL-2B in a side-by-side test (tracker se-066).

## Model preparation

`normalize` loads the raw upstream weights exactly as Transformers does (renames, ties,
casts, constructor buffers) and adds the upstream configuration and processor files that
ship in `character_describer/upstream/`. Inference never downloads anything.

```sh
cozy run ./character-describer/normalize \
  model.source=hf://Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a/model.safetensors.index.json \
  --source-profile source=as-is/1 --upload-to <account>/qwen3.5-9b --rental <name> --await
cozy model publish <account>/qwen3.5-9b --release 0.1.0 --lane bf16=sha256:<checkpoint> --default-lane bf16
```

The same recipe serves any Transformers image-text model whose upstream files replace
`upstream/`; that is how the alternatives were compared.
