# Long-form video

A segment is one H3 call. It takes the same fields as `ref2va_turbo`: free-text `prompt`,
`duration_s` (5–15, required) and optional `seed`. A `long_form` segment also accepts
the booleans `context_frames` (carry the previous picture) and `context_audio` (carry the
previous sound), both defaulting to `true`. The model only ever sees one segment's prompt,
so the workflow copies the global information into it.

## Request

- `references`: characters, backgrounds, audio and videos, shared by every segment.
- `clean_references` (default `true`): BiRefNet removes the background of each supplied
  character image once, and H3 receives it composited onto white. The cleaned image
  appears in the reference outputs and is reused for every segment; `false` sends it as
  supplied. Supplied backgrounds, audio and videos are used as supplied. References generated
  from descriptions keep their existing path.
- `describe_references` (default `true`): a vision-language model writes one sentence
  about each supplied character image that has no `description` (identity, hair, face,
  clothing and items; never the background, pose or style). It runs once, alongside
  background removal, and joins that character's subject definition in every segment.
  An authored description is kept as written. If describing fails, the run continues
  without it and reports a warning.
- `style`: look shared by generated reference images and every segment (optional).
- `overall_soundscape`: sound shared by every segment (optional).
- `non_diegetic_music`: score shared by every segment (default `N/A`).
  Each segment may override it with its own `non_diegetic_music` string. Omitted or
  `null` inherits the shared score; blank or `N/A` requests no score. An explicit
  segment field replaces an inline music section. Without that field, inline music
  takes precedence over the shared score. This applies to both long-form composers.
- `segments`: the ordered calls.
- `context_frames`: global picture window for `long_form` (0, 22, 39 or 56; default 22).
- `mode` (`turbo` or `standard`) and `steps` (standard only).

Both long-form composers default to Turbo: eight PDD evaluations per segment.
The top-level `mode` applies to every segment. `mode: standard` uses 30 steps by
default; its optional `steps` selects 30, 40 or 50. Omit `steps` with `mode: turbo`.
There is no per-segment sampling override.

Omit CLI `--gpus` for this workflow: each model then uses its supported GPU count.
BiRefNet uses one GPU; H3 can still use all eight on a supported eight-GPU machine.
An explicit `--gpus=N` applies to every GPU child, so a value greater than one
cannot serve BiRefNet's single-GPU operation.

```yaml
mode: turbo
segments:
  - duration_s: 6
    prompt: "<Rover> approaches <Bridge>."
  - duration_s: 6
    context_frames: false
    prompt: "Cut to <Mara> watching from the far side of <Bridge>."
  - duration_s: 6
    prompt: "<Mara> waves as <Rover> lands beside her."
```

Segment 2 is a clean picture cut over continuous sound: music, room tone and the rhythm
of speech carry on, the picture does not.

Unknown fields are ignored, not refused; a missing required field, such as a misspelled
`duration_s`, still refuses.

```json
{
  "style": "Soft watercolor animation with muted pastel colors.",
  "overall_soundscape": "A low hum of laboratory machinery fills the hall.",
  "references": [
    {"name": "Subject-4", "kind": "character", "description": "An adult woman in a white lab coat."},
    {"name": "Background", "kind": "background", "image": "./lab.png"}
  ],
  "segments": [
    {
      "duration_s": 6,
      "prompt": "summary:\n[reference generation] <Subject-4> explores.\n\ndetailed_description:\n[Shot 1] <Subject-4> is walking through the laboratory <Background>, exploring and looking around."
    }
  ]
}
```

## What the workflow does to each segment

Write prompts in MiniMax's [full-reference format](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
`overall_soundscape`, `non_diegetic_music`, plus `style` after `summary`. Any section may
be missing. Text before the first heading counts as the description. Headings match in any
case, with `_`, `-` or a space between words.

1. **References.** Name them as `<Name>` (any case; `_`, `-` and space alike). Each segment
   gets every background, and the characters, audio and videos it names. An audio reference
   whose description names a character (a voice) follows that character. Backgrounds come
   first, in the global order; then the rest in the order the text first names them, with
   videos last.
2. **Numbering.** `<Picture k>`, `<Audio k>` and `<Video k>` count each kind over that
   attached set. A token such as `<Video 2>` names the reference called "Video 2" if there
   is one, else the global second video; either way it is renumbered for the call, so a
   segment naming only `<Video 2>` sends it as its `<Video 1>`. A token that matches
   nothing stays as written, with a warning.
3. **Sections.** The workflow adds `subject_definitions` (one line per attached
   reference), `retention_analysis` (one line per attached reference, with the
   `[Shot N]` markers that name it), and the global `style` (when set),
   `overall_soundscape` and `non_diegetic_music`. Audio follows MiniMax's guide: a voice
   is `<Audio k> is the voice-timbre reference for <Name> (Sx).`, reusing the speaker ID
   the text gives that character (none if it gives none), with
   `<Audio k>: reference - <Name>'s voice follows <Audio k>'s timbre and delivery without
   copying the original signal.`; other audio defaults to `reference`. A reference's own
   `retention_analysis` replaces its default line, e.g. `fully_copy - ...` to reuse music.
   A `summary` that opens with `[task types]` gains `audio reference` or `audio reuse`.
   - Existing subject, retention, and soundscape sections gain the global content.
   - If the section is missing, it is added in the order above, under the canonical heading.
   - A body of `N/A` gives way to described global content in those sections.
   - A segment's own music and `style` are kept as written, unless its explicit
     `non_diegetic_music` field replaces the music section.
4. **Everything else** reaches the model byte for byte.

H3's ref2va needs a picture. A segment with no background and no named character
therefore receives every image and audio reference, never a video, with a warning.

## Action videos

A `kind: video` reference guides action: choreography, timing, a fight's beats. Like a
character, it goes only to the calls whose text names it, so one clip can guide one segment
or several. Per call: at most three clips, 2–15 s each, 15 s in all, square pixels. A clip
with sound also conditions the audio and takes the next `<Audio N>` after the call's audio
references; strip the track (`ffmpeg -an`) to guide motion alone. Each clip gains
`<Video k> (motion and choreography): attribute_transfer`, which transfers its movement and
keeps none of its people or setting; a `<Video k>` line of your own replaces it.
`clean_references` and `describe_references` leave videos alone. Direct `ref2va` and
`ref2va_turbo` take video references the same way.

Write the role where you name the subject, as MiniMax's guide does. Name clips anything;
`Video 1`, `Video 2`… read naturally:

```json
{
  "references": [
    {"name": "Subject 1", "kind": "character", "image": "./ann.png"},
    {"name": "Subject 2", "kind": "character", "image": "./ben.png"},
    {"name": "Dojo", "kind": "background", "description": "A wooden dojo at dusk."},
    {"name": "Video 1", "kind": "video", "video": "./kick.mp4"},
    {"name": "Video 2", "kind": "video", "video": "./throw.mp4"}
  ],
  "segments": [
    {"duration_s": 6, "prompt": "subject_definitions:\n<Subject 1>'s motion comes from <Video 1>.\n\ndetailed_description:\n[Shot 1] <Subject 1> attacks <Subject 2> in <Dojo> with the spinning kick of <Video 1>."},
    {"duration_s": 6, "prompt": "subject_definitions:\n<Subject 2>'s motion comes from <Video 2>.\n\ndetailed_description:\n[Shot 1] <Subject 2> blocks and throws <Subject 1> as in <Video 2>."}
  ]
}
```

Segment 2 sends `throw.mp4` as its `<Video 1>`. Attach files by path in the request, or:

```sh
cozy run fidika/minimax-h3/long_form --input fight.json \
  --asset references.3.video=kick.mp4 --asset references.4.video=throw.mp4 --await
```

## Unwanted speech

H3 has no negative prompt. It fills audio time that no section describes with invented,
often non-English speech. Prevent it in the text:

- Describe the concrete non-voice sounds in `overall_soundscape`: room tone, weather,
  footsteps, cloth, impacts, breathing. "Environmental ambience continues across
  segments." names no sound. In six-second clips with five seeds each, that line gave
  18.5% speech; a concrete description gave 0%.
- Where nobody should speak, write "walks silently, her lips closed". After a line, add
  "then closes her lips". Characters who never speak get no `(S1)` ID.
- Use `N/A` in `overall_soundscape` only for total silence.

## Writing segments that join well

From community chains ([H3 Motion Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context#writing-prompts-for-a-chain)):

- Describe the continuation of what is on screen. H3 renders contradictions as unions:
  if segment N ends on A and segment N+1 describes B and C, you get all three.
- To change setup, open the segment by holding the previous closing framing for about
  2 s with no dialogue (an airlock), then cut.
- Give every hold micro-motion: a breath, a weight shift, an eyeline change. A held
  framing with nothing happening renders as a freeze.
- Timecodes in a continuation land early by the context window. `duration_s` counts new
  time only, but H3 samples the context first: a beat written for 4.0 s lands at 3.08 s
  with context 22 (0.92 s early; 1.63 s at 39, 2.33 s at 56).
- Keep context 22. A 56-frame window samples 2.33 s of old frames in every segment,
  costing compute and new time.
- Draft in Turbo; render long final chains with `mode: standard`. Turbo dulls the audio
  and softens the picture, and each segment inherits the previous one's output, so the
  loss compounds down a chain.
- In very long pieces, restart continuity with `context_frames: false` at a natural
  scene or musical transition.

## References and execution

Each generated character reference uses one Qwen Image 2.1 call and returns exactly one image.
That image is a 16:9, 2 MP character design sheet on one plain white canvas, in three
panels: a full-body front view, a full-body back view, and a neutral front-facing
head-and-shoulders close-up that gives H3 the face in detail. Appearance, proportions
and clothing stay consistent across the panels. The authored character description is
included verbatim apart from surrounding whitespace. Background references stay one 1:1,
1 MP environment image.

The global `style` text also reaches every Qwen character and background reference
request. For example, `style: realistic` guides both reference images and H3 video
segments. The sheet layout does not prescribe an art style: references follow
the explicitly requested style, default to a realistic photographic look when no
style is specified, and use anime styling only when explicitly requested.

A reference has:

- `name` and `kind` (`character`, `background`, `audio` or `video`);
- its file: `image` (a character or background), `audio` or `video`;
- optional `description`, `retention_analysis` and `seed`.

A supplied image skips generation. Otherwise Qwen-Image-2.1 generates the image once
from the description. Names normalise: whitespace collapses and `<`/`>` are dropped. Two
names that match the same token are refused. Attach files with a JSON path, or with
`--asset references.0.image=...`, `--asset references.3.audio=...` or
`--asset references.4.video=...`. Audio references are 2–15 s each, 15 s in total, at most
three. Allowed: up to nine images and twelve image and audio references in all; videos count
per call. An audio-only set is refused.

```sh
cozy run fidika/minimax-h3/long_form --input story.json --await
```

## Continuation

Each segment after the first may carry two things from its predecessor, chosen separately:

| `context_frames` | `context_audio` | Result |
| --- | --- | --- |
| `true` (default) | `true` (default) | Motion and sound continue |
| `false` | `true` | Clean picture cut; music, room tone and speech rhythm continue |
| `true` | `false` | Motion continues; the sound starts fresh |
| `false` | `false` | A fresh start |

The picture is the previous segment's last `context_frames` frames (global: 22, 39 or 56;
0 carries no picture anywhere, while sound still follows each segment's `context_audio`).
The sound is always its last second. Both are the previous segment's own latents, never a
decoded and re-encoded copy, so colour and sharpness do not drift down a chain.

The carried picture becomes the literal opening of the new clip, so a prompt that opens
with a cut to other characters loses to it: H3 keeps the old characters. Use
`context_frames: false` for a cut; the run warns when a carrying segment's first shot cuts
to characters the previous segment's last shot does not name.

A segment delivers at least its `duration_s`: the native grid snaps it up, and nothing
sampled after the carried frames is discarded. A carried picture of 22/39/56 frames allows
at most 14/13/12 new seconds, and carried sound alone (a one-second head) 14; longer
segments are shortened and reported. `long_form_cuts`
renders independent segments. Turbo uses eight PDD evaluations, and standard allows
30/40/50 steps.

Outputs arrive as they are made. Prepared and generated reference images are published in
the request's reference order, each once it and every earlier one exist, and the video after
every segment as the joined film so far: one file,
`<run>-video.mp4`, appended in place (a fragmented MP4, one init plus one fragment per
segment) that `cozy run play <run>` follows live. The final MP4 is the last revision, byte
for byte, bounded at 256 MiB. A failed or canceled run fails, but keeps the references and
the film through its last good segment. Loudness is set per segment as it is joined, so a
later, louder segment never changes a fragment already published.
