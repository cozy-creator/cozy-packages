# Long-form video

Use `long_form` for continuation of one camera take and `long_form_cuts` for
independently generated segments. Both take the same authored segment content.

A **segment** is one model invocation. A **shot** is a camera shot inside that
segment. Every segment starts at `[Shot 1]`; segment 9 never becomes `[Shot 9]`.
Only write `[Shot 2] At 00:03.000, ...` inside a description when that segment
actually contains an intentional camera cut. Times are relative to that segment.

## Input

Declare one shared `style`, one to nine shared `references`, and a `segments` list.
Each segment requires `summary`, `detailed_description`, `overall_soundscape`, and
`non_diegetic_music`. Optional fields are `duration_s`, `seed`, `dialogue`, and
`screen_text`. Continuous segments default to 10 seconds; independent cut segments
retain the 15-second default. Durations must be 5–15 seconds before context adjustment.

```json
{
  "style": "Realistic live action with warm afternoon lighting.",
  "references": [
    {"name": "Mara", "kind": "character", "description": "An adult wearing a green coat."},
    {"name": "Garden", "kind": "scene", "description": "A garden with a stone path and wooden gate."}
  ],
  "segments": [
    {
      "summary": "{Mara} greets someone at the gate in {Garden}.",
      "detailed_description": "{Mara} (S1) stands beside the gate in {Garden} and says: <d>[English] Good morning.</d> She closes her mouth and walks along the path.",
      "overall_soundscape": "Leaves rustle and footsteps sound on the path.",
      "non_diegetic_music": "N/A",
      "duration_s": 10
    },
    {
      "summary": "{Mara} continues along the path in {Garden}.",
      "detailed_description": "The camera follows {Mara} walking through {Garden}.",
      "overall_soundscape": "Footsteps and rustling leaves continue.",
      "non_diegetic_music": "N/A",
      "duration_s": 10
    }
  ]
}
```

This is the 1.18.2 input contract. The earlier `shots`, per-item `prompt`, per-item
reference selections, and shared `soundscape`/`music` fields are replaced by this
explicit segment schema. `style` remains shared rather than copied into each input
segment. The output envelope still reports delivered/requested segment counts.

## Exactly what H3 receives

Every invocation gets these six sections, in this order, following the
[official reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md):

1. `subject_definitions`: the shared reference definitions.
2. `summary`: `[reference generation]` followed by that segment's authored summary.
3. `retention_analysis`: the shared visual-reference roles.
4. `detailed_description`: shared style, then `[Shot 1]`, then that segment's text.
5. `overall_soundscape`: that segment's ambience and physical sounds.
6. `non_diegetic_music`: that segment's audience-only score.

Definitions, retention analysis, reference image order, and Subject/Picture mapping
are identical in every segment. There are no per-segment reference subsets or
reordering. Character actions and dialogue use `<Subject N>`. `<Picture N>` appears
in the subject definitions to identify the source image. `{name}` placeholders
resolve to those fixed Subject labels, case-insensitively.

The compiler does not add camera/continuity/narration prose before the authored
scene. It does not repeat `[Shot 1]` when the description already begins with it.
The guide's headings and reference task label are literal model input. Progress
segment numbers, hashes, character counts, and report headings are not model input.

Blank audio strings remain blank; `N/A` explicitly requests complete silence for
soundscape or no audience-only score for music. Dialogue stays in the detailed
description, never in summary or the two audio sections.

## Dialogue and visible text

Use stable speaker IDs assigned by the order of first speech across the whole story:

```text
{Mara} (S1) says warmly: <d>[English] Good morning.</d>
```

`<d>` is the H3 speech delimiter, not `<speech>`. Place only the language and spoken
words inside it, with sentence punctuation before `</d>`. Delivery belongs outside.
Reference labels identify visual subjects; `(S1)` identifies a voice and remains
stable across segments. Do not put speaker IDs in retention analysis.

Alternatively, use a segment's structured `dialogue` list. Each line has `speaker`
(a shared character name), `language`, `text`, optional `delivery`, and optional
`voiceover`. Insert its one-based `{dialogue:N}` marker exactly once in the detailed
description. The compiler assigns global speaker IDs by actual marker order and
preserves the exact supplied words. A story uses either structured dialogue or manual
`<d>`/speaker markup, avoiding collisions. Voiceover uses the prescribed off-screen
phrase and a lips-closed direction.

Narrative quotation delimiters are removed, retaining apostrophes within words.
Manual `<d>` bodies and structured speech remain verbatim. For exact visible lettering,
put up to 16 strings in `screen_text` and insert each `{screen_text:N}` marker once:
`A sign reads {screen_text:1}.` The compiler quotes that literal after narrative
cleanup. Literal speech/display text is never rescanned for placeholders.

[The two-segment garden example](examples/dialogue-garden.json) demonstrates structured
speech, shared references and exact sign lettering. These formats control prompt
construction; intelligible speech and correct lettering still require output review.

## References and execution

A reference has `name`, `kind` (`character` or `scene`), optional `description`,
optional `image`, and optional `seed`. Without an image, a nonempty description is
required and Qwen-Image-2.1 generates the reference once. A supplied image skips
reference generation. Character generation adds a white-background reference portrait
direction only to Qwen; H3 receives the declared visual description.

Names start with a letter and use letters, digits, `_` or `-`; case-insensitive
names must be unique. All names, dialogue, segment fields and compiled prompt lengths
(maximum 4096 characters per model prompt) are validated before any reference generation.
PNG/JPEG/WebP image fields use the normal verified asset path, with 64 MiB encoded and
256 MiB decoded limits. Relative filenames resolve beside the input JSON; absolute
and `~/` paths work. HTTP URLs are not input assets. `--asset references.0.image=...`
is also supported; do not supply the same image twice by both mechanisms.

```sh
cozy package install paul/minimax-h3
cozy run paul/minimax-h3/long_form --input story.json --rental=your-rental --await
```

Use `long_form_cuts` in the command for independent camera setups. Outputs go to the
normal package-specific output directory. Only the assembled MP4 and final PNG are
downloaded; reference images, intermediate clips and continuation context stay under
ordinary retained child-result custody. A later-segment failure returns completed
media with `complete=false`; first-segment failure and cancellation remain terminal.

Turbo uses its trained eight PDD evaluations. `mode="standard"` permits 30, 40 or 50
steps. Reference generation uses 40 steps and 1024×1024 PNGs. There is no segment-count
cap; the assembled MP4 retains its 256 MiB bound. Installing code does not start
inference or download model weights. All generated segments use the same shared refs.

Continuous mode retains a completed audio/video tail. With `context_frames` 22, 39 or
56, later segments deliver at most 14, 13 or 12 new seconds respectively; requested
longer durations are shortened and reported. The first segment may deliver 15 seconds.
At 24 fps, default 10-second segments deliver 240 new frames each, without duplicating
the context frames. Context preserves rendering provenance, not arbitrary Python state;
there is no guarantee of seamless motion, identity or speech continuity.
