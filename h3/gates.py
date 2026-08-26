"""THE TWO PER-REQUEST QUALITY GATES (#524 tiers 1 and 2), over cozy-eval's own instruments.

se-002's first render was container-valid and content-garbage: two streams, the right
resolution, the right fps, a billing row — and moving noise with a noise soundtrack. Both
of the checks that would have caught it ALREADY EXISTED in cozy-eval (median adjacent-frame
correlation against a measured 0.6 floor; the absolute audio defect budget), and this
endpoint called neither. Its own local floor was a whole-clip standard deviation whose
POSITIVE FIXTURE was literally `torch.rand` — green for the exact statistical class of the
failure.

WHY IMPORT RATHER THAN VENDOR, adjudicated here rather than left implicit:

  * cozy-eval is the OWNER'S NAMED QUALITY GATE (#520, "you already have a tool for it"),
    and the numbers in it are CALIBRATED — the 0.6 correlation floor sits in a measured
    empty middle (VAE noise 0.29, real renders 0.92-0.99) and the six audio limits are
    fitted against a 41-clip known-good union with a 9-clip held-out population. A copy of
    a calibrated constant is a second authority that cannot be recalibrated.
  * It is ALREADY a dependency of the serve environment in practice: `cozy_eval` is
    installed in this repo's `.venv`, and `pyproject.toml` has carried `cozy_eval.*` in the
    optional-to-typecheck list since se-001 — the same treatment torch gets.
  * An ABSENT cozy-eval must REFUSE, never quietly skip. An import does that for free; a
    vendored copy would make the gate look present when its calibration is not.

The import is LAZY, inside the functions, for the same reason torch's is: `describe` runs
in a container with no GPU, no weights and no numpy, and cozy-eval brings numpy.

WHAT EACH TIER IS FOR, and what neither claims. Tier 1 reads the TENSORS the endpoint is
about to encode; tier 2 decodes the MP4 THAT WAS ACTUALLY WRITTEN. They are different
subjects — a correct tensor can still be muxed into a container with the wrong frame rate,
and that is precisely the 0.88 s A/V mismatch the first artifact carried. NEITHER IS A
QUALITY GATE: both catch structural invalidity and corruption, and cozy-eval's own scope
note says a melted render scores HIGHER on adjacent-frame correlation than a clean one.
Perceptual quality and prompt adherence are tiers 3-5 and live in cozy-eval proper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cozy_runtime.author import OutputError, Telemetry

from h3_arch.layout import MediaFacts

#: `nb_frames` is a container hint and some muxers leave it 0. A hint that disagrees with
#: the decoded count is the container lying, and the DECODED count is what a viewer gets.
_MISSING = 0


def _cozy_eval() -> Any:
    """cozy-eval or a REFUSAL. UNMEASURED is never a pass (cozy-eval's own tri-state law),
    so a gate that cannot run is a failed generation and not a skipped check."""
    try:
        import cozy_eval.audio as ce_audio
        import cozy_eval.frames as ce_frames
        import cozy_eval.integrity as ce_integrity
        import cozy_eval.metrics.audio as ce_metrics_audio
    except ImportError as exc:
        raise OutputError(
            "cozy-eval is not installed in this deployment, so the output gates cannot "
            "run: an unmeasured generation is not a passed generation, and this endpoint "
            f"does not publish media it has not checked ({exc})",
            code="output_gate_unavailable",
        ) from exc
    return ce_audio, ce_frames, ce_integrity, ce_metrics_audio


def _refuse(code: str, defects: list[str], what: str) -> None:
    if defects:
        raise OutputError(f"{what}: " + "; ".join(defects), code=code)


def pre_encode_gate(
    torch: Any,
    *,
    decoded: Any,
    pixels: Any,
    waveform: Any,
    requested: MediaFacts,
    tel: Telemetry,
) -> None:
    """TIER 1 — the tensors, before a single byte is encoded.

    `decoded` is the VAE's FLOAT video (the #411 lesson: NaN is read before quantization,
    because `clamp(0,1).to(uint8)` erases exactly the evidence a diverged decode leaves).
    `pixels` is the `(T, H, W, 3)` uint8 clip the encoder will receive. `waveform` is
    `(channels, samples)` float.
    """
    _, _, ce_integrity, ce_metrics_audio = _cozy_eval()

    # --- finite values, on the float decode
    video_nan = float((~torch.isfinite(decoded)).float().mean())
    audio_nan = float((~torch.isfinite(waveform)).float().mean())
    tel.metric("video_nonfinite_fraction", round(video_nan, 6))
    tel.metric("audio_nonfinite_fraction", round(audio_nan, 6))
    if video_nan > 0.0 or audio_nan > 0.0:
        raise OutputError(
            f"the decode produced non-finite values over {video_nan:.4%} of the video and "
            f"{audio_nan:.4%} of the audio, and this endpoint does not publish it: a "
            "non-finite decode is a failed generation, not a clip with artefacts",
            code="output_integrity_nan",
        )

    # --- exact shapes and counts, against what was REQUESTED
    shape = tuple(int(n) for n in pixels.shape)
    want = (requested.frames, requested.height, requested.width, 3)
    if shape != want:
        raise OutputError(
            f"the decoded clip is {shape} and the request resolved to {want}: a frame "
            "count or canvas that does not match the plan is a geometry defect upstream "
            "of the encoder, not a presentation choice",
            code="output_shape_mismatch",
        )
    channels, samples = int(waveform.shape[0]), int(waveform.shape[1])
    audio_seconds = samples / float(requested.sample_rate)
    tel.metric("audio_seconds", round(audio_seconds, 4))
    if not requested.av_agrees(audio_seconds):
        drift = requested.av_drift(audio_seconds)
        raise OutputError(
            f"the soundtrack is {audio_seconds:.3f} s against {requested.duration_s:.3f} s "
            f"of video ({drift:.3f} s apart): the two streams were denoised jointly and a "
            "duration disagreement means one of the two latent geometries is wrong",
            code="output_av_duration_mismatch",
        )

    # --- noise, blank and GRID, cozy-eval's calibrated floors over the real pixels
    integrity = ce_integrity.output_integrity(pixels.cpu().numpy())
    tel.metric("adjacent_frame_corr", round(integrity.adjacent_frame_corr or -1.0, 4))
    tel.metric("frame_std_min", round(integrity.frame_std_min or -1.0, 5))
    # REPORTED, not just gated. This axis exists because a render that PASSED the two floors
    # above at 0.988 was rejected on sight for a lattice (#557), so the number that catches
    # it belongs in the render's own record and not only in a refusal string.
    tel.metric("grid_peak_ratio", round(integrity.grid_peak_ratio or -1.0, 3))
    tel.metric("grid_period_px", round(integrity.grid_period_px or -1.0, 1))
    if not integrity.ok:
        raise OutputError(
            f"the decoded video fails cozy-eval's integrity floor — {integrity.summary()}",
            code="output_integrity_video",
        )

    # --- clipping, level and silence, cozy-eval's calibrated defect budget
    stats = ce_metrics_audio.signal_stats(
        waveform.to(torch.float32).cpu().numpy().T, requested.sample_rate
    )
    _report_audio(stats, channels=channels, tel=tel)


def post_encode_gate(path: Path, *, requested: MediaFacts, tel: Telemetry) -> None:
    """TIER 2 — the MP4 THAT WAS WRITTEN, decoded.

    Everything here is read out of the container by ffprobe and ffmpeg, because the
    container is what a caller opens. The first artifact passed every in-memory check the
    endpoint had and still shipped 103 frames of 4.29 s video against 5.18 s of audio."""
    ce_audio, ce_frames, _, ce_metrics_audio = _cozy_eval()

    width, height, fps, hinted = ce_frames.probe(path)
    tel.metric("encoded_width", float(width))
    tel.metric("encoded_height", float(height))
    tel.metric("encoded_fps", round(fps, 4))

    defects: list[str] = []
    if (width, height) != (requested.width, requested.height):
        defects.append(f"encoded {width}x{height}, requested {requested.width}x{requested.height}")
    if abs(fps - requested.fps) > 1e-3:
        defects.append(f"encoded {fps:g} fps, requested {requested.fps}")

    decoded_frames = sum(1 for _ in ce_frames.iter_video(path))
    tel.metric("encoded_frames", float(decoded_frames))
    if decoded_frames != requested.frames:
        defects.append(f"decoded {decoded_frames} frames, requested {requested.frames}")
    if hinted != _MISSING and hinted != decoded_frames:
        defects.append(f"the container claims {hinted} frames and {decoded_frames} decode")
    _refuse("output_container_video", defects, "the encoded video disagrees with the request")

    if requested.mute:
        tel.metric("encoded_audio_streams", 0.0)
        return

    rate, channels, _ = ce_audio.probe_audio(path)  # refuses when there is NO audio stream
    audio = ce_audio.read_audio(path)
    tel.metric("encoded_sample_rate", float(rate))
    tel.metric("encoded_channels", float(channels))
    tel.metric("encoded_audio_seconds", round(audio.duration, 4))

    video_seconds = decoded_frames / float(requested.fps)
    if rate != requested.sample_rate:
        defects.append(f"encoded {rate} Hz, requested {requested.sample_rate} Hz")
    # THE TWO TIERS BOUND DIFFERENT QUANTITIES, and this one is not the tensor's. Tier 1
    # already held the SAMPLES to one video frame; what is left for the container is
    # whether the mux lost or invented audio — and a block codec cannot express an
    # arbitrary length, so it pads its tail to a whole frame. The first full-length proof
    # render measured that live: 15.075 s of samples, 15.083 s of video (8 ms apart, tier 1
    # green) and 15.136 s in the container, which is 53 ms — outside a one-video-frame
    # tolerance and inside one AAC frame of it. A correct render was being refused.
    quantum = _codec_quantum(path, audio)
    tolerance = requested.av_tolerance_s + quantum
    tel.metric("audio_codec_quantum_s", round(quantum, 6))
    if requested.av_drift(audio.duration) > tolerance:
        defects.append(
            f"{audio.duration:.3f} s of audio against {video_seconds:.3f} s of video, "
            f"{requested.av_drift(audio.duration):.3f} s apart — more than one video frame "
            f"({requested.av_tolerance_s:.3f} s) plus one codec frame ({quantum:.3f} s)"
        )
    _refuse("output_container_audio", defects, "the muxed container disagrees with itself")

    _report_audio(ce_metrics_audio.signal_stats(audio), channels=channels, tel=tel)


def _codec_quantum(path: Path, audio: Any) -> float:
    """One audio CODEC FRAME, in seconds, READ OFF THIS CONTAINER.

    Derived, never a constant: the encoder is the runtime's choice, and AAC's 1024 samples
    is not AC-3's 1536 or Opus's variable frame. The container states how many coded frames
    it holds, so the quantum is its decoded length divided by that count — which is the
    number this file needs and the one `cozy_eval.audio.probe_audio` does not return.

    A container that does not state a frame count yields 0.0, which leaves the tolerance
    exactly where tier 2 had it before: the fallback tightens, never loosens.
    """
    import json
    import subprocess

    try:
        raw = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-select_streams", "a:0",
             "-show_entries", "stream=nb_frames", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
        frames = int(json.loads(raw)["streams"][0].get("nb_frames") or 0)
    except (OSError, ValueError, KeyError, IndexError, subprocess.CalledProcessError):
        return 0.0
    return audio.duration / frames if frames > 0 else 0.0


def _report_audio(stats: dict[str, float], *, channels: int, tel: Telemetry) -> None:
    """cozy-eval's absolute audio defect budget, applied and reported.

    The limits and their provenance are cozy-eval's `AUDIO_DEFECTS`; nothing is redefined
    here. Stereo-only limits are skipped for mono input rather than scored zero, which is
    the same distinction cozy-eval draws between "mono source" and "stereo that collapsed".
    """
    ce_audio, _, _, _ = _cozy_eval()
    for name in ("audio_rms_dbfs", "audio_true_peak_dbtp", "audio_clip_fraction",
                 "audio_max_silence_run", "audio_dc_offset"):
        if name in stats:
            tel.metric(name, round(stats[name], 6))

    breached: list[str] = []
    for defect in ce_audio.AUDIO_DEFECTS:
        if defect.metric == "audio_stereo_separation_db" and channels < 2:
            continue
        value = stats.get(defect.metric)
        if value is not None and defect.breached(value):
            breached.append(f"{defect.metric} {value:g} breaches {defect.describe()}")
    if breached:
        raise OutputError(
            "the soundtrack breaches cozy-eval's absolute audio defect budget — "
            + "; ".join(breached)
            + ": H3 generates audio jointly, so a broken track is a half-failed "
            "generation and not a muted preference",
            code="output_integrity_audio",
        )
