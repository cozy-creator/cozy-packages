# Source identity qualification

The focused source probe ran in the cached CPU worker image `sha256:88fe1703fd8e20c6987324459b6379989557b94bd83eccf619358464f80e04ec`, network disabled, 1 CPU and 4 GiB limit, against Runtime memo API source `5625dad5`. No native binaries were scanned or edited. The probe copies only package Python/data files and shadows installed distribution metadata in a temporary directory.

```json
{
  "qualified": "source-identity-only",
  "operations": [
    "apply_adaln",
    "assemble_full",
    "assemble_normalized",
    "compute_adaln_tables",
    "normalize_component",
    "prepare_turbo",
    "retable_adaln",
    "select_adaln_weights"
  ],
  "caller_stable": true,
  "sdxl_helper_invalidates": true,
  "sdxl_resource_invalidates": true,
  "h3_kernel_invalidates": true,
  "h3_resource_invalidates": true,
  "native_version_invalidates": true,
  "regular_cli_matrix_complete": false
}
```

The normal global CLI reuse and interrupted recovery matrix remains a separate qualification gate. Runtime >=0.18.30 is required; lock refresh awaits its public release.
