# SDXL latent and decoder diagnosis

This owned diagnostic copies the four actual installed SDXL2.4.0 source/assets
from ilulu generationec300577. SOURCE.json records every original hash. The
original model construction, tokenizer, conditioning, scheduler, denoise and
standard decode code is unchanged; AST-PROOF.json verifies its original helpers
and model methods. Public packages and existing environments/cache are untouched.
Python3.12 and all69 observed dependency versions are constrained; the reviewed
private Runtime/TensorFS pair remains supplied by the owned machine.

`generate` keeps the literal original1024-square20-step request and selected
paul/sdxl1.0.0 BF16 model. After denoise it copies the final `[1,4,128,128]` FP16
latent to CPU and publishes exactly131072 raw bytes before decoding. SHA256,
little-endian C-order NCHW geometry, actual SDK, original/diagnostic source hashes,
scheduler/scale and observed decoder input shapes accompany its image result.
The decoder hook observes shapes and returns no replacement. Native invoke modes
remain independent evidence. This extra output is diagnostic, not a scored timing.

`decode_latents` accepts that same exact saved file and its SHA256 and performs
one new decode operation. It checks length/hash, reconstructs the original FP16
latent and uses the original SDXL scale/upcast rule. `untiled` uses standard
decode on an adequate-memory owned worker and must show full128-square decoder
input plus untiled native mode. `tiled512` explicitly calls stock tiled_decode
with512-sample/64-latent tiles. No decoder is rerun after failure, no denoise is
repeated by these operations, and no request or saved latent is resized.

Reviewable execution plan, before any GPU launch:

1. Install this explicit local package through ordinary default-home `cozy package
   install PATH --editable`. Verify its two entrypoint schemas; fp8/mxfp8 jobs are
   copied original surface and are not invoked.
2. On ilulu with the existing fixed1.5GiB holder113/birth33024000, submit one new
   explicit diagnostic `generate` using original grouped-h index0 input and
   `model.model=paul/sdxl@1.0.0`. Save image, raw latent, JSON/source/native/holder
   receipts on the controller. Verify20 current steps and exact128KiB hash.
3. Root allocates one adequate-memory3070, estimated20–60min/$0.06–$0.17 at the
   recent total$0.1717/hour, or assigns an explicitly owned suitable idle worker.
   Import the exact saved latent as the declared FileAsset through ordinary CLI
   `--asset=latents=FILE`; independently run untiled and tiled512 with the same
   selected model/source and private SDK. Retain each operation's own id and mode.
4. Compare all four RGB outputs: original4281 vs diagnostic producer, independent
   untiled vs tiled512, each independent decode vs the preserved full-memory
   control. The35dB rule remains unchanged. This separates denoise drift from
   tiling without treating decoder reruns as recovery of an earlier forward.
5. Capture terminal/source/physical receipts and explicitly end only owned rentals
   once no accepted work remains. Keep the failed4280 and quality-failed4281.

Current proof: source hashes, unchanged AST and CPU-only application discovery
using the actual generated environment with CUDA hidden and bytecode writes
disabled. No diagnostic GPU request or adequate-memory allocation has started.
