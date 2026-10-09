# Long-form video

A segment is one H3 call. It takes the same fields as `ref2va_turbo`: free-text `prompt`,
`duration_s` (5–15, required) and optional `seed`. A `long_form` segment also accepts
the boolean `context_frames`, defaulting to `true`; `false` disables inherited motion for that segment. The model only ever sees one segment's
prompt, so the workflow copies the global information into it.

## Request

- `references`: characters, scenes and audio, shared by every segment.
- `style`: look shared by generated reference images and every segment (optional).
- `overall_soundscape`: sound shared by every segment (optional).
- `non_diegetic_music`: score shared by every segment (default `N/A`).
  Each segment may override it with its own `non_diegetic_music` string. Omitted or
  `null` inherits the shared score; blank or `N/A` requests no score. An explicit
  segment field replaces an inline music section. Without that field, inline music
  takes precedence over the shared score. This applies to both long-form composers.
- `segments`: the ordered calls.
- `context_frames`: global window for `long_form` (0, 22, 39 or 56; default 22).
- `mode` (`turbo` or `standard`) and `steps` (standard only).

Both long-form composers default to Turbo: eight PDD evaluations per segment.
Each segment accepts a boolean `turbo`, defaulting to `true`. Set `turbo: false`
for a standard 30-step segment, such as a shot with complex motion. Global
`mode: standard` takes precedence and makes every segment standard; its optional
`steps` still selects 30, 40 or 50. Omit global `steps` whenever any segment uses Turbo.
Strings and numbers are not valid segment `turbo` values.

```yaml
segments:
  - duration_s: 6
    prompt: "<Rover> approaches <Bridge>."
  - duration_s: 6
    turbo: false
    context_frames: false
    prompt: "<Rover> leaps across the gap at <Bridge>."
  - duration_s: 6
    prompt: "<Rover> rolls away from <Bridge>."
```

`turbo` and `context_frames` are independent. Switching samplers keeps the same
base checkpoint and ordinary references. When continuation is enabled, the next
segment receives the completed audio/video context from its actual previous
segment, including a previous segment rendered with the other sampler.

Unknown fields are ignored, not refused; a missing required field, such as a misspelled
`duration_s`, still refuses.

```json
{
  "style": "Soft watercolor animation with muted pastel colors.",
  "overall_soundscape": "A low hum of laboratory machinery fills the hall.",
  "references": [
    {"name": "Subject-4", "kind": "character", "description": "An adult woman in a white lab coat."},
    {"name": "Background", "kind": "scene", "image": "./lab.png"}
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
   gets every scene, the characters it names, and the audio it names. An audio reference
   whose description names a character (a voice) follows that character. Scenes come
   first, in the global order; then the rest in the order the text first names them.
2. **Numbering.** `<Picture k>` and `<Audio k>` count each kind over that attached set.
   A global `<Picture N>` or `<Audio N>` in the text also names its reference and is
   renumbered to match. A token that matches nothing, such as `<Video 1>`, stays as
   written, with a warning.
3. **Sections.** The workflow adds `subject_definitions` (one line per attached
   reference), `retention_analysis` (one line per attached reference, with the
   `[Shot N]` markers that name it), and the global `style` (when set),
   `overall_soundscape` and `non_diegetic_music`.
   - Existing subject, retention, and soundscape sections gain the global content.
   - If the section is missing, it is added in the order above, under the canonical heading.
   - A body of `N/A` gives way to described global content in those sections.
   - A segment's own music and `style` are kept as written, unless its explicit
     `non_diegetic_music` field replaces the music section.
4. **Everything else** reaches the model byte for byte.

H3's ref2va needs an image. A segment with no scene and no named character therefore
receives every reference, with a warning.

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

## References and execution

Each character reference uses one Qwen Image 2.1 call and returns exactly one image.
That image is a 16:9, 2 MP character design sheet on one plain white canvas, in three
panels: a full-body front view, a full-body back view, and a neutral front-facing
head-and-shoulders close-up that gives H3 the face in detail. Appearance, proportions
and clothing stay consistent across the panels. The authored character description is
included verbatim apart from surrounding whitespace. Scene references stay one 1:1,
1 MP environment image.

The global `style` text also reaches every Qwen character and scene reference
request. For example, `style: realistic` guides both reference images and H3 video
segments. The sheet layout does not prescribe an art style: references follow
the explicitly requested style, default to a realistic photographic look when no
style is specified, and use anime styling only when explicitly requested.

A reference has:

- `name` and `kind` (`character`, `scene` or `audio`);
- optional `description`, `image`, `audio`, `retention-analysis` and `seed`.

A supplied image skips generation. Otherwise Qwen-Image-2.1 generates the image once
from the description. Names normalise: whitespace collapses and `<`/`>` are dropped. Two
names that match the same token are refused. Attach files with a JSON path, or with
`--asset references.0.image=...` / `--asset references.3.audio=...`. Audio references are
2–15 s each, 15 s in total, at most three. Allowed: up to nine visual references and
twelve references in all. An audio-only set is refused.

```sh
cozy run fidika/minimax-h3/long_form --input story.json --await
```

`long_form` uses one global numeric `context_frames`, defaulting to 22. A segment's
`context_frames` defaults to `true`, which inherits that window. Set it to `false` to
disable continuation for that segment. Per-segment numeric values are refused. Global 0 disables
continuation throughout. The first segment always uses 0 because there is no previous clip.

Global 0 or a segment's `false` attaches no previous audio/video context. Ordinary authored character,
scene and media references still apply; the preceding clip is not turned into another
reference. This removes motion inheritance, but does not guarantee a visible cut.
Continuation can resume after an independent segment. With global 56 and only segment 3
set to `context_frames: false`, segment 2 continues from segment 1, segment 3 receives no
previous context, and segment 4 continues from segment 3. Each segment exports only the
window its successor needs, and exports none when that successor opts out or the global value is 0.

`context_frames` 22/39/56 allows at most 14/13/12 new seconds for that segment;
0 allows 15 seconds. Longer segments are shortened and reported. `long_form_cuts` renders
independent segments. Turbo uses eight PDD evaluations, and standard allows 30/40/50
steps.

Outputs arrive as they are made. Each generated reference image is published when it
lands, and the video after every segment as the joined film so far: one file,
`<run>-video.mp4`, appended in place (a fragmented MP4, one init plus one fragment per
segment) that `cozy run play <run>` follows live. The final MP4 is the last revision, byte
for byte, bounded at 256 MiB. A failed or canceled run fails, but keeps the references and
the film through its last good segment. Loudness is set per segment as it is joined, so a
later, louder segment never changes a fragment already published.
