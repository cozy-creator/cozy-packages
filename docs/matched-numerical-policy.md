# Matched Euler numerical policy

Tracker #319/#324. Owner: Codex benchmark_execution. Source base: packages-v2
444ea624d4bf87fad372ac4fd96fd1f9108c79f8. These are new production candidates;
installed SDXL2.4 and Anima0.2.16, their outputs and controls remain historical.
No GPU inference or release is claimed by this plan.

SDXL currently generates CUDA FP16 noise, multiplies by Diffusers' leading-grid
`init_noise_sigma`, computes CFG in FP16 and rounds state to FP16 every step.
Stock ComfyUI20ca544 generates CPU FP32 noise, scales a below-maximum first sigma
by sigma, converts denoiser outputs to FP32 and keeps Euler state in FP32.
The actual20-step grids differ by4.77e-6, but initial noise scales differ by
0.04524994. Matching just the grid does not match the authored computation.

Anima's Diffusers0.40 blocks begin with FP32 latents but compute CFG in BF16;
FlowMatchEulerDiscreteScheduler then returns model-output dtype, reducing state
to BF16 after its first step. Stock Comfy keeps state and CFG in FP32.
The default Anima block also casts its timestep to BF16 before dividing by1000;
stock Comfy passes the normalized FP32 sigma into the sinusoidal embedding.

The new candidates use a request-local CPU torch generator seeded directly from
the authored seed. Noise is FP32 and transfers once to the runtime-selected
device. SDXL follows stock EPS initial scaling: sigma for a below-maximum start,
sqrt(1+sigma²) for a start at the model maximum. Denoiser inputs stay in each
model's actual compute dtype (SDXL FP16, Anima BF16). CFG and scheduler updates
use FP32, and the next state remains FP32. Dimensions, steps, prompts, negative
prompts, guidance and modifiers retain their authored meaning. Versioned output
provenance states the new policy; there is no hidden benchmark-only switch.
Anima's narrow maintained-block subclass preserves FP32 timestep values before
embedding, without changing the BF16 denoiser input.

The implementation retains Diffusers' scheduler and supported Anima block
interfaces. CPU proof executes pinned Comfy symbols and actual Diffusers0.40
against shared synthetic predictions: initialization, full20/30-step schedules,
CFG-enabled/disabled paths, dtype at every step, and extreme finite prediction
values. Source-equivalent floating expressions need a recorded error bound;
CPU proof cannot establish full-network or image equality. The new packages
must receive their own three repeated unconstrained controls and all constrained
quality cells, on the exact coherent machine/Runtime/TensorFS candidate.

CONFIRMED CPU: `scripts/matched-numerical-proof.py` passes24 checks against actual
Torch2.14/Diffusers0.40 and Comfy20ca544. This includes real production block
assembly and the SDXL denoiser-input method, eight exact CPU noise cases,
EPS scaling below/at/above maximum, all30 Anima timestep/embedding inputs, and
twelve20/30-step synthetic prediction schedules. Worst scaled differences from
stock FP32 Euler are7.72e-7 SDXL and5.10e-7 Anima. The deliberately large finite
prediction red arm overflows the old SDXL FP16 state; it is not a real-model
failure claim. Actual full-network conditioning, image quality and GPU timings
remain open. Candidate versions are SDXL2.7.0 and Anima0.3.1.

Decode strategy is recorded separately. A tile strategy may differ across
engines only if each engine passes the same predeclared fidelity rule against
its own independent unconstrained reference. Matching decode precision remains
required. Neither an adaptive tile nor a saved nonflat image proves quality.
Current1.5GiB tiled SDXL30dB failure remains unqualified.

Source evidence:
- [Comfy noise and dtype](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/sample.py)
- [Comfy EPS initialization](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/model_sampling.py)
- [Comfy Euler and CFG](https://github.com/Comfy-Org/ComfyUI/blob/20ca544ee0436721d8eb5f544665e490609f72c8/comfy/samplers.py)
- [Diffusers Euler precision](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_euler_discrete.py)
- [Diffusers Anima blocks](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/modular_pipelines/anima/denoise.py)
