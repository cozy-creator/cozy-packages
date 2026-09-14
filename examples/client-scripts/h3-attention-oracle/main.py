# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime[media,minimax-h3]>=0.18.2,<1", "minimax-h3"]
# [tool.uv.sources]
# minimax-h3 = {path = "../../../minimax-h3", editable = true}
# ///
"""Run actual-input attention comparisons as an ordinary unpublished client script.

Requires the paired Runtime development artifact described in README.md. The
request supplies Input fields such as prompt, backends and capture_step.
"""

from cozy_runtime.author import Context, Outputs, Telemetry
from h3_attention_oracle import Input, OracleModel, Result, probe

from h3 import KeyframeAssets


def main(
    ctx: Context, *, payload: Input, model: OracleModel, out: Outputs, tel: Telemetry
) -> Result:
    return probe(ctx, payload, KeyframeAssets(), model, out, tel)
