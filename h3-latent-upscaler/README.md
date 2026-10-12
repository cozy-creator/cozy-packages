# H3 learned latent upscaler

Runtime owns this model's construction, placement and BF16 weights. The attributed
LBH-123-AI 3D Conv v1 network enlarges normalized H3 video latents spatially, preserving
time and leaving audio untouched. Target latent dimensions are exact, from 1x to 4x.
The H3 package owns the follow-up refinement and video decode.

The upstream ComfyUI node normalizes raw VAE latents first. Diffusers H3 latents are
already normalized with those same per-channel mean/std values; this adapter calls
the network directly and returns normalized CPU float32 latents. Full temporal context is the default. Optional temporal chunking uses the upstream
32-row chunks with overlap blending; it is not numerically identical to a full-context
forward because GroupNorm and temporal convolutions see different windows.

Code: MIT, upstream revision `40316cf008b2fd8663263270669eb4da23f89d2c`.
Weights: Apache-2.0, Hugging Face revision `3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`.
See `h3_latent_upscaler/upstream/NOTICE` and the retained licenses.

The CPU normalize job adds configuration/order and preserves the imported weight bytes:

```sh
cozy run ./h3-latent-upscaler/normalize model.source=hf://LBH-123-AI/Minimax_h3_latent_Upscaler@3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors --source-profile source=as-is/1 --upload-to paul/h3-latent-upscaler --tensorhub local --await
```
