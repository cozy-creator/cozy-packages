"""Path-free integrity checks over the exact tensors handed to Runtime's encoders."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from cozy_runtime.author import OutputError, Telemetry


@dataclass(frozen=True, slots=True)
class MediaFacts:
    width: int
    height: int
    frames: int
    fps: int
    sample_rate: int
    mute: bool

    @property
    def duration(self) -> Fraction:
        return Fraction(self.frames, self.fps)

    @property
    def av_tolerance(self) -> Fraction:
        return Fraction(1, self.fps)

    def av_agrees(self, audio_seconds: Fraction) -> bool:
        return abs(audio_seconds - self.duration) <= self.av_tolerance


def _cozy_eval() -> tuple[Any, Any, Any]:
    try:
        import cozy_eval.audio as ce_audio
        import cozy_eval.integrity as ce_integrity
        import cozy_eval.metrics.audio as ce_metrics_audio
    except ImportError as exc:
        raise OutputError(
            "cozy-eval is absent, so this package cannot prove its generated tensors",
            code="output_gate_unavailable",
        ) from exc
    return ce_audio, ce_integrity, ce_metrics_audio


def pre_encode_gate(
    torch: Any,
    *,
    pixels: Any,
    waveform: Any,
    video_nonfinite_fraction: float,
    audio_nonfinite_fraction: float,
    requested: MediaFacts,
    tel: Telemetry,
) -> None:
    """Refuse malformed, non-finite or blank/noisy generated tensors.

    Audio defect metrics are emitted as telemetry and never refused.
    """
    ce_audio, ce_integrity, ce_metrics_audio = _cozy_eval()

    if waveform.ndim != 2:
        raise OutputError(
            f"the audio decode has shape {tuple(waveform.shape)}, expected (channels, samples)",
            code="output_audio_shape",
        )
    tel.metric("video_nonfinite_fraction", round(video_nonfinite_fraction, 6))
    tel.metric("audio_nonfinite_fraction", round(audio_nonfinite_fraction, 6))
    if video_nonfinite_fraction or audio_nonfinite_fraction:
        raise OutputError(
            "the official H3 decode produced non-finite video or audio values",
            code="output_integrity_nan",
        )

    shape = tuple(int(value) for value in pixels.shape)
    expected = (requested.frames, requested.height, requested.width, 3)
    if shape != expected:
        raise OutputError(
            f"the generated pixel tensor is {shape}, expected {expected}",
            code="output_shape_mismatch",
        )
    channels, samples = (int(value) for value in waveform.shape)
    if channels not in (1, 2):
        raise OutputError(
            f"the generated soundtrack has {channels} channels, expected mono or stereo",
            code="output_audio_channels",
        )
    audio_duration = Fraction(samples, requested.sample_rate)
    tel.metric("audio_seconds", round(float(audio_duration), 4))
    if not requested.av_agrees(audio_duration):
        raise OutputError(
            f"the generated soundtrack is {float(audio_duration):.3f}s against "
            f"{float(requested.duration):.3f}s of video",
            code="output_av_duration_mismatch",
        )

    integrity = ce_integrity.output_integrity(pixels.cpu().numpy())
    tel.metric("adjacent_frame_corr", round(integrity.adjacent_frame_corr or -1.0, 4))
    tel.metric("frame_std_min", round(integrity.frame_std_min or -1.0, 5))
    tel.metric("grid_peak_ratio", round(integrity.grid_peak_ratio or -1.0, 3))
    tel.metric("grid_period_px", round(integrity.grid_period_px or -1.0, 1))
    if not integrity.ok:
        raise OutputError(
            f"the generated video fails cozy-eval's integrity floor: {integrity.summary()}",
            code="output_integrity_video",
        )

    stats = ce_metrics_audio.signal_stats(
        waveform.to(torch.float32).cpu().numpy().T, requested.sample_rate
    )
    for name in (
        "audio_rms_dbfs",
        "audio_true_peak_dbtp",
        "audio_clip_fraction",
        "audio_max_silence_run",
        "audio_dc_offset",
    ):
        if name in stats:
            tel.metric(name, round(stats[name], 6))

    # Cozy-eval's AUDIO_DEFECTS budget is REPORTED, never refused. cozy-eval itself
    # demoted it to report-only after it falsely rejected real content, and a prompt like
    # "an empty room, silence, then a distant door slam" legitimately exceeds
    # `audio_max_silence_run high=0.25`. Refusing here destroyed a complete, billed
    # 345-frame generation over a soundtrack the request asked for.
    for defect in ce_audio.AUDIO_DEFECTS:
        if defect.metric == "audio_stereo_separation_db" and channels < 2:
            continue
        value = stats.get(defect.metric)
        if value is not None and defect.breached(value):
            tel.metric(f"{defect.metric}_breached", 1)
