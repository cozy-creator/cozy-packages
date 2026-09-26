# Qualification

The initial source checks left installed packages unchanged. The integrated proof below
uses a separate local package alias in the normal Creator home.

- Strict mypy: h3.py and story.py, two files, zero errors using the existing private Runtime .18.28/TensorFS .54 typing environment.
- Actual BasedPyright language server: story.py strict diagnostics captured in story-lsp.json; explicit list annotations remove prior unknown member diagnostics.
- Model-free async resolver proof executes the production resolve_reference_images function: all supplied images call the generator zero times; mixed input preserves the exact supplied handle and generates only missing images; a generation failure cancels and drains its sibling.
- Prompt proof executes production functions: mixed-case names and placeholders, duplicate-case refusal, missing-description refusal, image-plus-description inclusion, six official sections, style before Shot 1, and distinct first/continuation wording.
- Static source interface regenerated with installed private Runtime .18.28 and parsed successfully (36,910 bytes).
- Creator PR 695 verifies filenames become standard nested asset bindings. The normal global CLI now includes this change and exercised it in run1127.
- Duration planning uses the native MAX_FRAMES budget minus context before flooring to seconds; all 5-15s requests become admissible windows. The synthetic long-form proof expectations now cover 15+12+12 seconds with 56-frame context.

## Integrated GPU proof, 2026-09-26

Normal global Creator CLI, default home: run1127 / job-30ce275193c1ac37673633c0,
local/minimax-h3-mixed-check/long_form, root-owned Reishi (2x H100 SXM80).
The candidate h3.py/story.py are byte-identical to source commit7dfb84e.
Run1125/1126 failed before execution because the private test copy's package.toml
still named an experimental entrypoint; correcting it to h3:app fixed the test setup.

- JSON supplies Mara as an existing PNG by relative filename and Garden as a generated scene.
- Logs confirm Mara source=attached with the original digest, Garden source=generated.
  Only Garden invokes Qwen. Mixed-case reference placeholders succeed.
- New style/soundscape/music fields and the six-section compiler execute successfully.
- First shot delivers5s. The requested15s second shot shortens to12s with56 context frames.
- Complete=true, delivered2, delivered_frames408, fps24; the result contains the shortening warning.
- ffprobe:1344x768,24fps,17.032s muxed audio/video. Exactly final MP4 and PNG are exported
  under ~/.cozy/outputs/local-minimax-h3-mixed-check; both sizes and SHA256 digests verified.
- Final PNG was visually inspected. This is an integration proof, not a controlled quality comparison.
- Rental explicitly ended through cozy rental end; live rental list confirms Reishi absent.

Evidence: cozy_v2/outputs/wire62-video-typed-cohort-20260926/mixed-check-qualified-{result.json,events.jsonl}.
Input: cozy_v2/r/h3-mixed-input-qualification-20260926/request.json.
Video digest:656db506b9553a267aca4a2044640ba0fb37d76fdc77efa2f6a35e6d44b6b6fd.

Not claimed: public publication, release-lock refresh, a full synthetic broker run, or a
visual-quality guarantee. The user's installed local/minimax-h3 and existing JSON remain unchanged.
