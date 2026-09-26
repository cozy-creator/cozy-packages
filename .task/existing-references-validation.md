# Isolated qualification

No installed CLI, daemon, package, rental or active source candidate was changed.

- Strict mypy: h3.py and story.py, two files, zero errors using the existing private Runtime .18.28/TensorFS .54 typing environment.
- Actual BasedPyright language server: story.py strict diagnostics captured in story-lsp.json; explicit list annotations remove prior unknown member diagnostics.
- Model-free async resolver proof executes the production resolve_reference_images function: all supplied images call the generator zero times; mixed input preserves the exact supplied handle and generates only missing images; a generation failure cancels and drains its sibling.
- Prompt proof executes production functions: mixed-case names and placeholders, duplicate-case refusal, missing-description refusal, image-plus-description inclusion, six official sections, style before Shot 1, and distinct first/continuation wording.
- Static source interface regenerated with installed private Runtime .18.28 and parsed successfully (36,910 bytes).
- Creator PR 695 separately verifies filenames become standard nested asset bindings. Its global CLI is not installed.
- Duration planning uses the native MAX_FRAMES budget minus context before flooring to seconds; all 5-15s requests become admissible windows. The synthetic long-form proof expectations now cover 15+12+12 seconds with 56-frame context.

Not claimed: a new real Creator-to-worker mixed-image invocation, GPU inference, full synthetic broker proof, visual quality, publication, or release-lock refresh. The predecessor PR 276 still owns the private continuous-workflow integration. This source deliberately changes future input fields without modifying the current working installation.
