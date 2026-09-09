# SDXL pre-measurement policy proposal

Status: **PROPOSED, NOT RATIFIED**. Authored for se-042 on 2026-09-09 before the
reference/repeat/candidate renders. No numerical SDXL measurements or prior approval
are claimed. `conditions.proposed.json` is ordinary `cozy-eval/gate-config@2` data;
`workloads.json` is the existing Workload data, and `checklists.json` uses Eval's
existing authored checklist format. There is no new workflow format.

Eight distinct prompts cover object/color fidelity, counting, a human scene,
animal motion, a landscape, transparent/metallic materials, watercolor and illustration.
Each has three explicit VQA items; no generated checklist or unbound OCR model is
needed. Seeds5031–5038,20steps, guidance7,1024², no HiDiffusion, fixed negative prompt.
Every prompt renders candidate, reference and same-seed reference repeat through the
ordinary nonmemoized SDXL entrypoint. UNet capture steps0,10,19 are requested; only
teacher-forced step0 supplies the binding activation gate, later steps report divergence.

## Proposed limits and their meaning

These are engineering hypotheses for calibration, not borrowed H3 bands:

| Category | Proposed condition | Rationale to test |
| --- | --- | --- |
| Weights | worst relative Frobenius ≤0.05 | A5% per-tensor reconstruction ceiling should separate a small quantization perturbation from the required ×2-scale red arm; actual SDXL sensitivity is unmeasured. |
| Activations | promote≤4×repeat; reject>8×repeat; rel-L2 cap0.10 | Compare against the measured same-seed floor, with a10% absolute backstop; the multipliers and cap need held-out calibration. |
| Image signals | imaging_index0.90–1.10; imaging_worst_prompt≥0.80 | Propose a10% population detail/tonal allowance and20% worst-prompt allowance; these are image-feature ratios, not a human quality scale. |
| Paired outputs | PSNR-distance mean≤4×repeat, worst≤8×repeat | An in-run sensitivity screen only; stochastic/trajectory changes must not be reinterpreted as semantic damage. |
| Quality | worst element_recall≥0.80 | With three equally weighted items this requires all three; test whether the actual bound Qwen judge agrees with independent human labels. |

Saturation is **not asserted as zero**. Quantizer-time clipping is not recoverable
from decoded values alone, and Eval must preserve its absence until a legitimate
measurement exists. No saturation-dependent policy may pass with an invented counter.
LPIPS, ARNIQA, MUSIQ and other unbound learned metrics remain unmeasured; they are not
silently dropped from a policy that names them. The proposal uses only available
managed measurement surfaces plus the explicitly selected owner-held Qwen instrument.

A zero repeat floor with a nonzero candidate delta remains `no_floor`/INDETERMINATE
under Eval's existing contract. Do not insert epsilon, cache the repeat, relax the
floor, or pretend the cap alone ratified an absolute activation gate. A deterministic
zero floor may require a separately reviewed policy/API change before admission.

## Calibration required before ratification

1. Pin source/reference/candidate identities, SDXL package/wheel closure, worker image,
   Runtime/Eval/instrument versions, GPU/driver and exact authored data digests.
2. Qualify the learned judge on independently labeled positive/negative images. A
   successfully prepared Qwen checkpoint is not evidence of scoring calibration.
3. Run null/reference-repeat controls on the same worker and run the proposed clean
   candidate. Record every missing metric and actual floor, not only verdict strings.
4. Run the real ×2-weight-scale red candidate through the same entrypoint. Also bank
   authored missing-object/wrong-color examples and blur/flat-output controls. They
   must expose the intended failures; pixel edits do not substitute for the weight arm.
5. Calibrate using separate data or leave-one-out evaluation. Never use a population's
   own fitted bands as its admission proof. If a null control fails, investigate the
   metric/policy rather than declaring the baseline broken or silently widening bands.
6. Review and freeze a new conditions digest before the held-out candidate assessment.
   Every adjustment is a new policy/version and retains the old observations. Record
   who ratified it, the evidence paths/digests and scope of its validity.

The client keeps provisional measurements/reports local and publication disabled.
`approved_conditions` must exactly match an explicitly ratified digest before checkpoint,
report attachment or release effects are allowed. A numerical verdict under this proposal
is a provisional screen, not admission and not a claim of prior ratification.
