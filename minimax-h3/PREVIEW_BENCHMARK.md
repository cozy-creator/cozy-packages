# Private matched preview experiment

Run 2004 completed the requested comparison on the user-provided Asirpa rental,
`pr-15975cbca908e1b3ef8f`, through the ordinary default-home Cozy CLI.

- A: generate completed AV latents at 960×480, apply the learned spatial upscaler
  to 1536×768, then decode through the H3 VAEs. There is no second denoising pass.
- B: generate completed AV latents at 1536×768, then decode through the same VAEs.

Both arms used four NVIDIA H100 80GB HBM3 GPUs with the same actual GPU UUIDs,
Sol attention dense and SageAttention sparse choices, model/LoRA manifests,
Ref2VA Turbo 8-step schedule, prompt, exact character/background PNG bytes and
1024×1024 normalized references. The upscaler used one H100. Both returned
240 frames at 24 fps and 320,000 stereo audio samples: 10 delivered seconds.
Both canvases have a 2:1 aspect ratio; this is not a replay of run 1937's 1344×768.

The request seed was 2768991793. The actual Torch generator seed was
4073474745462816035 in both arms. The receipt stores those exact decimal digits
as a string because canonical JSON cannot carry a full-width uint64 integer.
Resolution changes the video noise grid and later audio RNG draws, so matching
seeds cannot promise the same take. A preserves its source audio and temporal
latent grid while applying the learned spatial lift.

## Measured result

| Handler stage | A: 480p then latent upscale | B: native 768p |
| --- | ---: | ---: |
| Prepare | 0.167 s | 0.121 s |
| Condition | 5.384 s | 3.335 s |
| Denoise, including completed CPU readback | 18.354 s | 46.462 s |
| Retain generated latents | 0.047 s | 0.169 s |
| Learned latent upscale | 0.975 s | — |
| Decode and encode | 16.935 s | 5.744 s |
| Sum of measured handler stages | 41.862 s | 55.830 s |

Denoising was 2.53× faster at 480p in this pair. A paid first-use decoder work;
B reused the warmed decoder. Outside these handler clocks, A waited 1430.437 s
behind the user's workflow and paid 128.455 s of child model/executor preparation.
B paid 10.893 s of preparation. Arm wall time excluding A's queue was 172.569 s
versus 67.729 s for B. This ordered pair does not establish a fair warm wall-time
speedup. CPU source preparation was cached on the final successful transaction.

Both videos follow the walking/look-around prompt and resemble the red-dressed
character. Room framing, camera perspective and gaze timing differ. At matched
four-second frames, native face/eye detail is cleaner; the learned-upscaled result
shows more jagged eye detail and stronger purple makeup-like marks. This is a
single sample, not a general quality verdict.

The final directory contains exactly three ordinary CLI products:

- `~/.cozy/outputs/h3-preview-comparison/2004-preview_upscaled.mp4`
- `~/.cozy/outputs/h3-preview-comparison/2004-native.mp4`
- `~/.cozy/outputs/h3-preview-comparison/2004-metadata.json`

GStreamer reports 10 seconds and seekable video for both. Five cold/non-keyframe
seeks per file match sequential decoded pixels. The SDK trims AAC padding to the
exact 320,000-sample audio endpoint. Diagnostic frames and receipts stay private.

The GPU run succeeded, but automatic collection exposed a generic Runtime
`Products.finish` bug: journaling the first of multiple final products released
later recipient holds. Both videos were recovered read-only from their exact
remote TensorFS blobs, hash-verified, and placed at the recorded CLI paths. The
ordinary Creator observer then verified the files and finished with `collected=1`.
No records or retention checks were bypassed, and neither successful arm was
repeated for collection. The generic Runtime fix is tracked independently.

## Frozen inputs and qualification

The private SDK remains at Runtime PR #1013 source
`0b69e9939aed93bf5d826b5150504a47980ce401`, version
`0.18.92+dev.heea81b1ca46c39b7f97613465f0bae339eea76aca8b2898ec16e25d4839dcb6b`.
Its exact native wheel SHA256 is
`c78b396962670ff4d9eb8a8a4780a81bba991ea0456190443f906940f95a85a8`.
Embedded source-wheel provenance SHA256 is
`50916f77cf41182b2f9fbc67538517321a6a0f8e3fe97239f9463ef92e5038e2`.
These are different artifacts; neither was relabeled or rebuilt during this run.
Remote CPU software probe 1992 verified the installed SDK, explicit canvas support
and CUDA untouched. Worker Runtime was 0.18.93; CLI/daemon were 0.1.21.

The upscaler is the pinned FP16 safetensors at HF
`LBH-123-AI/Minimax_h3_latent_Upscaler@3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`,
member `minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors`.
Its 690,592,672 bytes have SHA256
`043e5a48e161610ef6c3ea974645220354d06fa618abca15f76d084812eb55c2`.
Its 322 tensors total 345,280,216 parameters. Architecture and normalization are
from Director 63834b9f8561aa14a6e5289ac3d20dee78bb7d23; upstream licenses are included.
Normal Runtime source operations acquire and convert the carrier; native writer
capabilities derive FP32 weights/config. Model, Loader and uses_components own
residency. No larger media caps or package-level HTTP/manual CUDA loading is used.

CPU proofs cover exact upstream geometry, the entire checkpoint census, bitwise
small-network 1.6× parity, managed model scope, retained AV custody, actual native
writer derivation, all eight managed broker routes and real 10-second MP4 encoding.
The broker regression now uses production canonical serialization and an exact
full-width Torch seed, reproducing the original private receipt failure before
its fix. Both reference files also have explicit image admission bounds.

Private failed run 1993 and user-canceled run 1999 remain recorded. The user asked
to move workflow 2003 ahead; fresh transaction 2004 honored that normal queue
order. No prior cancellation was undone and no replacement rental was created.
This branch remains a private experiment with no public upscaler API/publication.
