# /// script
# requires-python = ">=3.12"
# dependencies = ["torch==2.14.0", "pillow>=12"]
# ///
"""Qwen3.6's expert routing and decode speed on this machine's GPU (se-085).

    cozy run ./examples/client-scripts/qwen3.6-routing.py model.model=<qwen3.6-35b-a3b or 27b release> \
        picture=./picture.png request=./request.json --on <machine>

`request` is `{"prompt": ..., "thinking": false, "max_tokens": 1024}`. One question about `picture`,
answered by the model's own `ask`, with every MoE layer traced: experts hit per layer and token,
the working set, and the hit rates of an LRU and a next-use (MIN) expert cache of each size in
`CACHES`, MIN's misses per forward being the fewest experts that must move. A dense model reports
no routing. The answer, its tokens and its seconds come back too; lm's `[lm]` log line has the
decode rate.
"""

from __future__ import annotations

import io
import json
import time

from PIL import Image

from cozy_runtime import moe
from cozy_runtime.author import Context, Telemetry
from cozy_runtime.vlm import Question, VisionLM

#: Expert caches replayed, in experts: 3.15 MB each at int8, 1.77 MB at 4-bit (35B-A3B); 5,400 int8
#: experts is about what a 24 GB card keeps beside the rest.
CACHES = (1024, 2048, 4096, 5400, 8192)


def main(ctx: Context, model: VisionLM, tel: Telemetry, picture: bytes, request: bytes) -> str:
    asked = json.loads(request)
    routing = moe.Routing(experts=256, caches=CACHES)
    moe.Experts.trace = routing
    try:
        image = Image.open(io.BytesIO(picture)).convert("RGB")
        started = time.perf_counter()
        (answer,) = model.ask(
            [Question(asked["prompt"], [image])],
            system="",
            thinking=bool(asked.get("thinking", False)),
            stop=[],
            max_new_tokens=int(asked.get("max_tokens", 1024)),
            presence_penalty=0.0,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        seconds = time.perf_counter() - started
    finally:
        moe.Experts.trace = None
    tokens = len(model.processor.tokenizer.encode(answer, add_special_tokens=False))
    return json.dumps(
        {"answer": answer, "answer_tokens": tokens, "seconds": round(seconds, 2), "routing": routing.summary()}
    )
