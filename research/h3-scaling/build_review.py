"""Build a local review gallery from validated campaign artifacts, without copying videos."""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    review = campaign / "review"
    review.mkdir(exist_ok=True)
    artifacts = []
    posters = review / "posters"
    posters.mkdir(exist_ok=True)

    def relative(path: Path) -> str:
        path.resolve().relative_to(campaign)
        return os.path.relpath(path, review)

    def append(row: dict, video: Path, validation: Path, expected: str) -> None:
        actual = hashlib.sha256(video.read_bytes()).hexdigest()
        if actual != expected.removeprefix("sha256:"):
            raise ValueError(f"video hash changed: {video}")
        poster = posters / f"{actual}.jpg"
        if not poster.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-threads", "2", "-ss", "1",
                            "-i", str(video), "-frames:v", "1", "-vf", "scale=672:-2",
                            "-q:v", "4", "-an", "-threads", "2", str(poster)],
                           capture_output=True, check=True)
        row.update(video=relative(video), validation=relative(validation), sha256=actual,
                   bytes=video.stat().st_size, poster=relative(poster))
        artifacts.append(row)

    for hardware, directory, degrees in (("rtx5090", "rtx5090-8", (1, 4, 8)),
                                         ("h100", "h100-4", (4,))):
        for degree in degrees:
            for mode, steps in (("turbo", 8), ("regular", 30)):
                for repetition in (1, 2):
                    folder = campaign / directory / f"native-g{degree}-{mode}" / f"timed-{repetition}"
                    receipt = folder / "validation.json"
                    if not receipt.exists():
                        continue
                    data = json.loads(receipt.read_text())
                    if data.get("validation") != "pass" and data.get("status") != "passed":
                        continue
                    video = Path(data.get("video_path", folder / "outputs" / f"{data['run']}-video.mp4"))
                    append({"hardware": hardware, "degree": degree, "mode": mode,
                            "steps": steps, "engine": "Cozy", "variant": "default",
                            "repetition": repetition, "run": data["run"],
                            "execution_seconds": data["execution_ms"] / 1000,
                            "timing_boundary": "Recorded worker execution; transfer excluded",
                            "attention": ("Sage3 NVFP4 dense + Kitchen INT8 Sol" if hardware == "rtx5090"
                                          else "SageAttention dense + BF16 Sol"),
                            "attention_path": ("single-GPU shared-QKV producer" if hardware == "rtx5090" and degree == 1
                                               else "materialized head-local" if hardware == "rtx5090"
                                               else "BF16 Sol"),
                            "text_encoder": "FP8 rowwise"}, video, receipt, data["video_sha256"])

    ledger_path = campaign / "comfy" / "results-ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    for data in ledger.get("cases", []):
        if data.get("qualified") is not True or data.get("role") != "timed":
            continue
        receipt = Path(data["validation_file"])
        video = Path(data["video"]["path"])
        if not video.exists():
            continue
        append({"hardware": data["gpu"], "degree": data["gpu_count"],
                "mode": "turbo" if data["steps"] == 8 else "regular", "steps": data["steps"],
                "engine": "ComfyUI", "variant": data["configuration"],
                "repetition": data["repetition"], "run": data["prompt_id"],
                "execution_seconds": data["server_execution_seconds"],
                "timing_boundary": "Comfy server execution; observer transfer excluded",
                "attention": "Kitchen INT8 dense + Kitchen INT8 Sol",
                "text_encoder": "Official BF16 checkpoint; loader reports FP16", "capture": data["case"]},
               video, receipt, data["video"]["sha256"])

    inputs = []
    for repetition, label in ((1, "A · Fighting"), (2, "B · Racing")):
        payload = json.loads((campaign / "native" / f"input-turbo-{repetition}.json").read_text())
        inputs.append({"repetition": repetition, "label": label,
                       "seed": payload["seed"], "prompt": payload["prompt"]})
    index = {"generated_at": datetime.now(timezone.utc).isoformat(),
             "geometry": {"frames": 362, "width": 1344, "height": 768, "fps": 24,
                          "audio_hz": 32000, "audio_channels": 2},
             "review_status": "Human quality review pending for these new outputs",
             "inputs": inputs, "artifacts": artifacts,
             "planned_gpu_degrees": {"rtx5090": [1, 4, 8], "h100": [4]},
             "skipped_configurations": [{"hardware": "h100", "degree": 8,
                                         "reason": "Skipped at the user's request"}],
             "comfy_ledger": relative(ledger_path) if ledger_path.exists() else None,
             "comfy_ledger_updated_unix_s": ledger.get("updated_unix_s"),
             "comfy_scope_exceptions": ledger.get("scope_exceptions", []),
             "notes": ["Prompt and seed match between configurations for each selected input.",
                       "Input labels identify review samples, not distinct benchmark configurations.",
                       "Cozy uses the current rc.3 FP8 text encoder. Comfy loads the official BF16 checkpoint; its loader reports FP16, which alone does not prove every operand's compute dtype.",
                       "Single-GPU defaults use the shared-QKV producer; multi-GPU defaults use current-call materialized attention. These are different implementation paths.",
                       "Video/audio format and hash checks passed. They do not establish perceptual quality or equivalence.",
                       "Only validated timed captures appear. Missing Comfy cells remain placeholders until generated.",
                       "Baseline Comfy is primary until the benchmark owner selects a qualified alternative."]}
    (review / "artifact-index.json").write_text(json.dumps(index, indent=2) + "\n")
    (review / "artifact-index.js").write_text("window.BENCHMARK_ARTIFACTS = " + json.dumps(index).replace("</", "<\\/") + ";\n")
    (review / "index.html").write_text(Path(__file__).with_name("review.html").read_text())
    print(json.dumps({"gallery": str(review / "index.html"), "artifacts": len(artifacts),
                      "native": sum(a["engine"] == "Cozy" for a in artifacts),
                      "comfy": sum(a["engine"] == "ComfyUI" for a in artifacts)}))


if __name__ == "__main__":
    main()
