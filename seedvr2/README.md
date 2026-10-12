# SeedVR2 video restoration

Managed SeedVR2 3B restoration with the upstream Apache-2.0 architecture, one-step
sampler, fixed conditioning, standard PyTorch SDPA and Runtime-owned DiT/VAE weights.
`upscale` takes a video, resolution (`1080p`, `1440p`, `2160p`) and deterministic seed.
Frame presentation timestamps and audio are passed through Runtime's streaming media API.
Temporal windows overlap and use the upstream Numz crossfade.

This implementation is under qualification. CPU constructor/header parity, native
model normalization, GPU execution, package publication and benchmark results are
separate gates; this source alone does not claim any of them complete.

Weight source: `numz/SeedVR2_comfyUI` revision
`09ced71023636e9bc8cdf9cdecfb2625d1e691e8`, Apache-2.0 FP16 conversions of the
ByteDance-Seed model. Upstream provenance and modifications are in
`seedvr2/upstream/NOTICE`.
