# H3 learned latent upscaler

Runtime owns this model's construction, placement and BF16 weights. The attributed
LBH-123-AI 3D Conv v1 network enlarges normalized H3 video latents spatially, preserving
time and leaving audio untouched. Target latent dimensions are exact, from 1x to 4x.
The H3 package owns the follow-up refinement and video decode.

The network expects H3's normalized latents normalized again by the same per-channel
mean/std, as the ComfyUI node applies them to the latents it receives; this adapter does
the same around the network and returns H3-normalized CPU float32 latents. On real H3
latents the network's output, area-downsampled, has cosine 0.89 with its input this way
and about 0 without the second normalization. Full temporal context is the default. Optional temporal chunking uses the upstream
32-row chunks with overlap blending; it is not numerically identical to a full-context
forward because GroupNorm and temporal convolutions see different windows.

Code: MIT, upstream revision `40316cf008b2fd8663263270669eb4da23f89d2c`.
Weights: Apache-2.0, Hugging Face revision `3f941d5d182014dd5c0a5e16330420ee2d4aa0c6`.
See `h3_latent_upscaler/upstream/NOTICE` and the retained licenses.

The CPU normalize job adds configuration/order and preserves the imported weight bytes:

```sh
cozy run ./h3-latent-upscaler/normalize model.source=hf://LBH-123-AI/Minimax_h3_latent_Upscaler@3f941d5d182014dd5c0a5e16330420ee2d4aa0c6/minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors --source-profile source=as-is/1 --upload-to paul/h3-latent-upscaler --tensorhub local --await
```
