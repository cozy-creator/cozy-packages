# Character describer

`describe_character` describes one supplied character image in one sentence: identity, hair,
face, clothing and carried or worn items, never the background, pose or style, with greedy
decoding. A caption that names a setting or pose word anyway is regenerated once with those
words pointed out, and any clause that still names one is dropped. It takes one PNG, JPEG or
WebP image of up to 64 MiB. Describe several characters with one call each: TensorD runs them
on all of the run's GPUs at once. MiniMax-H3 uses it to fill a supplied character's missing
description.

```sh
cozy run fidika/character-describer/describe_character --asset image=./character.png --await
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

## The int8 lane

`int8` writes every projection and MLP of the 32 language layers as `int8-convrot256/1`
(W8A8 on the int8 tensor cores). The checkpoint drops from 17.5 to 11.1 GiB, so it fits a
16 GB card; the vision tower, embeddings and LM head keep their bytes.

```sh
cozy model quantize <account>/qwen3.5-9b@0.1.0/bf16 --int8 --rental <name> --await
cozy model publish <account>/qwen3.5-9b --release 0.1.0 --lane int8=sha256:<checkpoint>
```

