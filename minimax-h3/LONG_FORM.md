# Long-form video

H3 receives ordinary authored text. Cozy assembles the six section headings and
prepends the shared style; it does not interpret character tags, replace placeholders,
expand dialogue markers, remove quotes, insert camera shots, or rewrite sentences.
For example, authored `<Reina>` reaches the model as `<Reina>`.

## Overall fields and segment fields

The overall input contains:

- `style`: one shared overall-style string.
- `overall_soundscape`: optional ambience/physical sound shared by every segment.
- `non_diegetic_music`: optional audience-only score shared by every segment.
- `references`: the ordered image inputs or descriptions used to generate them.
- `segments`: the ordered generation requests.

Each segment requires `detailed_description` and `duration_s`. `summary`,
`overall_soundscape`, and `non_diegetic_music` are optional and default to blank.
When an overall audio field and its segment field are both supplied, the overall text
is prepended to the segment text in that section.
`seed` is optional. There are no per-segment reference subsets or reordered image slots.

```json
{
  "style": "Realistic live action with warm afternoon lighting.",
  "overall_soundscape": "A quiet room tone continues throughout.",
  "non_diegetic_music": "N/A",
  "references": [
    {"name": "Reina", "kind": "character", "description": "An adult woman wearing a green coat."},
    {"name": "Garden", "kind": "scene", "description": "A garden with a stone path and wooden gate.", "retention-analysis": "fully_preserved - the garden's gate and plants remain consistent."}
  ],
  "segments": [
    {
      "summary": "[reference generation] <Reina> greets someone at the gate in <Garden>.",
      "detailed_description": "[Shot 1]\n<Reina> (S1) stands beside the gate in <Garden> and says: <d>[English] Good morning.</d> She closes her mouth and walks along the path.",
      "overall_soundscape": "Leaves rustle and footsteps sound on the path.",
      "non_diegetic_music": "N/A",
      "duration_s": 10
    }
  ]
}
```

The package generates `subject_definitions` and `retention_analysis` from the ordered
references. Each reference can optionally provide `retention-analysis`; when absent,
the package uses its default role template. These generated sections are identical in
every segment prompt and are not user-level top-level fields.

The official [MiniMax reference guide](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md)
uses `<Subject N>` labels. Friendly labels such as `<Reina>` are an authoring experiment,
passed through literally; their generation quality is not established here. To use the
recommended labels, change the literal names consistently in definitions, retention,
summaries and descriptions. Cozy performs no conversion in either direction.

`references` configures image assets independently of the authored prompt text. Image
order determines `<Picture 1>`, `<Picture 2>`, etc. The package connects each literal
reference name to its generated subject definition. A reference's `name` identifies
its asset/configuration and seed; it is not a prompt placeholder.

## Exact model-facing assembly

For each segment, the package assembles:

```text
subject_definitions:
(shared subject_definitions text)

summary:
(this segment's summary text)

retention_analysis:
(shared retention_analysis text)

detailed_description:
(shared style text)
(this segment's detailed_description text)

overall_soundscape:
(this segment's overall_soundscape text)

non_diegetic_music:
(this segment's non_diegetic_music text)
```

The parenthesized lines above explain placement; they are not inserted into the model
prompt. Supplied field values, including tag names, quotes, punctuation and internal
whitespace, are preserved. If style is empty, only the segment description is used.
Blank summary/audio values are omitted. `N/A` audio values are omitted from the model
prompt entirely, so the text encoder does not spend tokens encoding an empty instruction.
Headers are separated by one blank line. Definitions,
retention text, style and ordered reference assets are shared identically across segments.

The package does not invent `[reference generation]`, `<d>`, `(S1)`, or `[Shot 1]` tags.
Write those ordinary H3 conventions directly in the corresponding text fields. Unknown
label text is not rejected or substituted. There is no custom brace/marker syntax.

## Segments and camera shots

A **segment** is one model invocation. A **shot** is a camera shot inside that segment.
Write `[Shot 1]` at the beginning of each segment's authored description. Segment 9
still begins with `[Shot 1]`, not `[Shot 9]`. The package never derives camera-shot
numbers from segment ordinals. For a cut within a segment, write for example
`[Shot 2] At 00:03.000, the camera cuts closer.` Times are relative to that segment.

One model prompt is assembled per segment. Report headings such as Segment 9,
progress counters and diagnostic hashes are not model input.

## Dialogue and visible text

Use H3's own dialogue format, not a Cozy-specific marker:

```text
<Reina> (S1) says warmly: <d>[English] Good morning.</d>
```

Put only the language label and spoken words inside `<d>...</d>`, ending complete
sentences with punctuation. Put delivery instructions outside the tags. Keep a speaker's
ID stable across the whole video; do not put speaker IDs in retention analysis. Place
quoted visible lettering directly in the visual description. Cozy preserves it verbatim.

Dialogue belongs in `detailed_description`. `overall_soundscape` describes ambient and
physical sounds; `non_diegetic_music` describes the audience-only score. H3's `N/A`
convention requests total silence in soundscape or no score in music. No automatic
quote detection, speech extraction, translation or punctuation repair is performed.

## References and execution

A reference has `name`, `kind` (`character`, `scene`, or `audio`), optional `description`,
optional `image`, optional `audio`, optional `retention-analysis`, and optional `seed`.
A supplied image skips generation. Without an image, a
nonempty description is required and Qwen-Image-2.1 generates that reference once.
Character generation adds its white-background portrait direction only to Qwen.
H3's textual definitions are generated from `references`; optional per-reference
`retention-analysis` customizes only that reference's generated retention line.

Audio references use a typed audio attachment and skip Qwen image generation:

```json
{
  "name": "ReinaVoice",
  "kind": "audio",
  "description": "It is the voice-timbre reference for <Reina> (S1), containing a spoken English vocal layer.",
  "audio": "./audio/reina.wav",
  "retention-analysis": "reference - its vocal timbre guides the dialogue delivery of <Reina> without copying the original signal."
}
```

The generated definition is `<Audio 1> is the supplied audio reference for <ReinaVoice>.`
followed by the description, so the description names the speaker it voices. In the segment,
write for example `Using the voice timbre referenced from <Audio 1>, <Reina> (S1) says: ...`.
An audio reference needs at least one character or scene reference in the same request;
audio-only reference sets are refused before any reference or model work starts.

Use `--asset references.3.audio=/absolute/path/reina.wav` instead of the JSON filename
when preferred. Audio references are bounded to 256 MiB encoded and 2 GiB decoded,
2–15 seconds each, 15 seconds aggregate, and up to three audio references. Visual
references remain capped at nine, with twelve total references. `<Picture N>` and
`<Audio N>` numbering is independent and follows the shared reference order. Audio
labels and supplied text are passed to H3; Cozy does not transcribe or rewrite them.

One to twelve references are supported: at most nine visual references and three audio
references. Asset names start with a letter and contain only
letters, digits, `_` or `-`, and must be unique ignoring case. PNG/JPEG/WebP image fields
use the normal verified asset path (64 MiB encoded, 256 MiB decoded). Relative filenames
resolve beside the input JSON; absolute and `~/` filenames also work. URLs are not input
assets. `--asset references.0.image=...` is supported as an alternative to an image field.

```sh
cozy package install fidika/minimax-h3
cozy run fidika/minimax-h3/long_form --input story.json --rental=your-rental --await
```

`long_form` continues completed audio/video context; `long_form_cuts` renders independent
segments.

Every segment states `duration_s` (5–15 seconds); there is no default. In continuous mode, context_frames 22/39/56 permits at
most 14/13/12 new seconds after the first segment. Longer requests are shortened and
reported. At 24 fps, a 10-second delivered segment has 240 new frames, without replaying
the context frames. Turbo uses eight PDD evaluations; standard mode permits 30/40/50 steps.
Reference generation uses 40 steps and 1024×1024 PNGs. Prompt validation happens before
reference generation; every complete model prompt is limited to 4096 characters.

Outputs use the default package folder. Only the assembled MP4 and final PNG are
downloaded; references, intermediate clips and continuation context remain retained
child results. The assembled MP4 retains its 256 MiB bound. Later-segment failure returns
completed media with complete=false; first-segment failure and cancellation stay terminal.
No prompt format guarantees speech intelligibility, identity preservation or seamless motion.
