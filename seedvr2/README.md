# SeedVR2 video restoration

Managed SeedVR2 3B restoration with the upstream Apache-2.0 architecture, one-step
sampler, fixed conditioning, standard PyTorch SDPA and Runtime-owned DiT/VAE weights.
`upscale` takes a video, resolution (`1080p`, `1440p`, `2160p`), deterministic seed,
and `aspect_ratio` (`source` by default, or `16:9` for an exact letterboxed canvas).
Frame presentation timestamps and audio are passed through Runtime's streaming media API.
Temporal windows overlap and use the upstream Numz crossfade.

This implementation is under qualification. CPU constructor/header parity, native
model normalization, GPU execution, package publication and benchmark results are
separate gates; this source alone does not claim any of them complete.

Weight source: `numz/SeedVR2_comfyUI` revision
`09ced71023636e9bc8cdf9cdecfb2625d1e691e8`, Apache-2.0 FP16 conversions of the
ByteDance-Seed model. Upstream provenance and modifications are in
`seedvr2/upstream/NOTICE`.

The normalize callable takes `model.dit`, `model.vae`, and a `conditioning` Tree
containing the upstream `pos_emb.pt`/`neg_emb.pt` constants. It verifies their exact
published hashes, reads them with PyTorch `weights_only=True`, and writes managed
BF16 tensors. The installed package carries no model files and inference performs
no file loading or upstream download.

Source ingestion and normalization use ordinary Cozy commands:

```sh
cozy model upload 'hf://numz/SeedVR2_comfyUI@09ced71023636e9bc8cdf9cdecfb2625d1e691e8/seedvr2_ema_3b_fp16.safetensors' paul/seedvr2-dit-raw --source-profile=as-is/1 --tensorhub local --await
cozy model upload 'hf://numz/SeedVR2_comfyUI@09ced71023636e9bc8cdf9cdecfb2625d1e691e8/ema_vae_fp16.safetensors' paul/seedvr2-vae-raw --source-profile=as-is/1 --tensorhub local --await
cozy run ./seedvr2/normalize model.dit=paul/seedvr2-dit-raw#sha256:DIT_CHECKPOINT model.vae=paul/seedvr2-vae-raw#sha256:VAE_CHECKPOINT conditioning=conditioning --input-tree conditioning=/path/to/upstream/constants --upload-to=paul/seedvr2 --tensorhub local --await
```

Resolution sets the short edge and preserves aspect ratio. Internally, padding
meets the model's 16-pixel grid and is cropped from the result. Audio is decoded
and re-encoded by Runtime with its source clock; it is not copied byte-for-byte.
Seventeen-frame windows share five frames and use upstream temporal blending.

`aspect_ratio=16:9` outputs exactly 1920×1080,2560×1440 or 3840×2160 using Pillow
letterboxing. The source subject is never stretched.
