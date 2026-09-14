# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime[media,minimax-h3]>=0.18.2,<1", "minimax-h3"]
# [tool.uv.sources]
# minimax-h3 = {path = "../../../minimax-h3", editable = true}
# ///
"""Run actual-input attention comparisons as an ordinary unpublished client script.

Requires the paired Runtime development artifact described in README.md. Edit
REQUEST before capture to select the prompt, backends and interception point.
"""

from cozy_runtime.author import Context, Outputs, Telemetry
from h3_attention_oracle import Input, OracleModel, Result, probe

from h3 import KeyframeAssets

REQUEST = Input(
    prompt=(
        "A continuous cinematic wide shot of two martial artists fighting in a rain-soaked "
        "courtyard. They exchange rapid punches, parries, spinning kicks and acrobatic dodges, "
        "moving around pillars while their clothing and splashing water follow each motion."
    ),
    duration_s=15,
    capture_step=10,
    capture_block=49,
    backends=("fa3_bf16", "sage2_sm90", "kitchen-int8", "sol-attn"),
)


def main(ctx: Context, *, model: OracleModel, out: Outputs, tel: Telemetry) -> Result:
    return probe(ctx, REQUEST, KeyframeAssets(), model, out, tel)
