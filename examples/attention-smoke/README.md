# Attention smoke before model loading

This ordinary private package loads no checkpoint. Run it only on an explicitly
owned rental after checking the signed bootstrap GPU UUID, GPU count, image digest
and Runtime source. It refuses an unexpected device, source, dependency version or
missing backend. It uses Runtime's own numerical kernel admission checks.

For development, stage a copy with the exact local Runtime wheel in
`tool.uv.sources`, then `uv lock` and install that copy through ordinary Creator.
Keep the default Cozy home so the requests remain visible:

```sh
cozy package install /path/to/staged/attention-smoke --editable --no-model-download
cozy run local/attention-smoke/probe --rental=OWNED_NAME --in smoke-input.json
```

The input JSON contains the verified `expected_runtime_source`,
`expected_gpu_uuid`, `expected_sm` (90 for H100), and `expected_visible_devices`.
Default H100 backends are FA3 BF16, Sage2, Kitchen INT8 and Sol. For Blackwell,
request `sdpa`, `flash-attn4` and `flashinfer-bf16-fp8` explicitly instead.

Use a normal serving entrypoint for this model-less GPU probe. A plain script's
`main()` can be admitted as CPU orchestration. An explicit request attention
override also has no model attention sites here; the probe itself resolves each
named backend and runs its numerical check.

`preparation_seconds` includes imports, first-call compilation and the tiny
numerical check. It is not an H3 benchmark or a quality comparison. A successful
smoke is permission to proceed with the separately measured full-shape and video
qualification, not permission to promote an approximate backend automatically.
