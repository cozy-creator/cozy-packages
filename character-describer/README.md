# Character describer

`describe_character` describes one supplied character image in one sentence: identity, hair,
face, clothing and carried or worn items, never the background, pose or style, with greedy
decoding. A caption that names a setting or pose word anyway is regenerated once with those
words pointed out, and any clause that still names one is dropped. It takes one PNG, JPEG or
WebP image of up to 64 MiB. Describe several characters with one call each: TensorD runs them
on all of the run's GPUs at once. MiniMax-H3 uses it to fill a supplied character's missing
description.

```sh
cozy run fidika/character-describer/describe_character --asset image=./character.png
```

The default model is Qwen3.5-9B (`qwen3.5-9b@0.2.0`), chosen over Qwen3.5-4B,
JoyCaption Beta One and Qwen3-VL-2B in a side-by-side test (tracker se-066). Its token loop
declares `min_bits=4, phase="decode"`: it serves the release's `nvfp4` lane where there is one.

`describe_places` describes supplied places the same way: one sentence about the place's look,
people left out. `describe_clip` watches a motion clip as video, the way Qwen-VL reads one:
about 2 frames a second, at least 16 and at most 32, at most 448 px long, each block stamped with
its `<t seconds>`. It answers in four parts, `PEOPLE`, `BEATS` (who does what to whom, which way,
facing the camera how, with times), `CAMERA` and `CUTS`, naming people only as P1, P2. ffmpeg's
scene score finds the cuts first (a frame scoring 0.2 or more starts a shot; on 15 test clips that
found all 10 cuts and no false one). They fill `CUTS`, and the model watches the whole clip once,
told where the shot changes and that people may differ across shots. Without ffmpeg
(`cozy tools install ffmpeg`; worker images bake it) the run warns and the model judges the cuts
itself. A 15 s clip is 30 frames and about 5 s to describe on an RTX A5000. `write_texts` answers text prompts,
all in one pass. Every entrypoint is memoized: the same file and instruction return the stored
answer without running the model.

```sh
cozy run fidika/character-describer/describe_clip --asset video=./fight.mp4
```

## Model preparation

The served checkpoint is the plain upload: TensorFS keeps the repository's `config.json` as the
construction config and its tokenizer, processor and chat template as assets. Parameters load in
their stored dtype (Qwen3.5 keeps `A_log` and its gated norms in f32). Inference never downloads
anything.

```sh
cozy model upload hf://Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a <account>/qwen3.5-9b \
  --on <name>
cozy model publish <account>/qwen3.5-9b --release 0.2.0 --lane bf16=sha256:<checkpoint>
```

## The int8 lane

`int8` writes every projection and MLP of the 32 language layers as `int8-convrot256/1`
(W8A8 on the int8 tensor cores). The checkpoint drops from 17.5 to 11.1 GiB, so it fits a
16 GB card; the vision tower, embeddings and LM head keep their bytes.

```sh
cozy run ./character-describer/int8 model.source=<account>/qwen3.5-9b@0.2.0/bf16 \
  --on <name>
cozy run upload <run> <account>/qwen3.5-9b
cozy model publish <account>/qwen3.5-9b --release 0.2.0 --lane int8=sha256:<checkpoint>
```

