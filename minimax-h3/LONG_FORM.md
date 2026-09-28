# Long-form video

A segment is one H3 call. It takes the same fields as `ref2va_turbo`: free-text `prompt`,
`duration_s` (5–15, required) and optional `seed`. The model only ever sees one segment's
prompt, so the workflow copies the global information into it.

## Request

- `references`: characters, scenes and audio, shared by every segment.
- `overall_soundscape`: sound shared by every segment (optional).
- `non_diegetic_music`: score shared by every segment (default `N/A`).
- `segments`: the ordered calls.
- `context_frames` (long_form), `mode` (`turbo` or `standard`) and `steps` (standard only).

```json
{
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
`overall_soundscape`, `non_diegetic_music`. Any section may be missing. Text before the
first heading counts as the description. Headings match in any case, with `_`, `-` or a
space between words.

1. **References.** Name them as `<Name>` (any case; `_`, `-` and space alike). Each segment
   gets every scene, the characters it names, and the audio it names. An audio reference
   whose description names a character (a voice) follows that character. Scenes come
   first, in the global order; then the rest in the order the text first names them.
2. **Numbering.** `<Picture k>` and `<Audio k>` count each kind over that attached set.
   A global `<Picture N>` or `<Audio N>` in the text also names its reference and is
   renumbered to match. A token that matches nothing, such as `<Video 1>`, stays as
   written, with a warning.
3. **Sections.** The workflow adds four sections: `subject_definitions` (one line per
   attached reference), `retention_analysis` (one line per attached reference, with the
   `[Shot N]` markers that name it), and the global `overall_soundscape` and
   `non_diegetic_music`.
   - If the segment already has one of these sections, the global content is appended
     inside it.
   - If the section is missing, it is added in MiniMax order, under the canonical heading.
   - A body of `N/A` gives way to described global content.
   - A global `N/A` never replaces a segment's own music.
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

`long_form` continues each segment from the previous segment's audio and video. In that
mode, `context_frames` 22/39/56 allows at most 14/13/12 new seconds after the first
segment; longer segments are shortened and reported. `long_form_cuts` renders
independent segments. Turbo uses eight PDD evaluations, and standard allows 30/40/50
steps. Only the assembled MP4 is downloaded. If a later segment fails, the completed part
is returned with `complete=false`.
