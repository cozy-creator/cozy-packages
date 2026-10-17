# H3 Turbo direct activation

Turbo is lightx2v's 8-step LoRA (`lightx2v/Minimax-h3-Turbo@a29e57d1`): the fl2v 8-step v1.0 file
for `fl2va`, the ref2v 8-step v1.0 768p file for `ref2va`, `ref2va_turbo` and long-form `mode: turbo`.
Each is rank 128, alpha 8, on the six attention and MLP families. The sampler is the ordinary
Euler loop at eight evaluations; video shift 12 (FL2VA) or 6 (Ref2VA), audio shift 3, no CFG.

The `minimax-h3-lightx2v` overlay holds each trunk's factors and the base model's AdaLN tables at
its turbo timesteps (the pruned base carries only the 30/40/50-step rows). The base heads serve
unchanged. `OfficialH3TurboLoRA.overlay()` returns the prepared component for the requested trunk;
`H3TurboLoRA._sample()` arms it for one request through `attention_kwargs` and restores the
caller's value in `finally`. The weight plane bakes the factors into the base's resident bytes.
