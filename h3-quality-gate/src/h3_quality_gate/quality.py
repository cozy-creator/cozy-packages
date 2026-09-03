"""job-003 — the H3 video/audio quality release gate, as a bounded job.

A PURE metric job over PAIRED publication manifests. It loads no serving Model, holds no
residency and renders nothing: the media it judges was produced by other attempts and
arrives as a granted tree, bound to those attempts by digest. That is the whole point —
a gate that could render its own evidence could also, quietly, render evidence that
passes.

Four v1 recoveries are structural here, not advisory:

  * **the verdict is HUMAN and tri-state.** `free_win | conditional_parity | reject`
    arrives on the payload or it does not exist. This job proposes nothing and infers
    nothing. `conditional_parity` classifies UNMEASURED and never auto-serves, and a
    candidate whose measured gate FAILED cannot carry a passing stamp — offering one
    refuses rather than recording it.
  * **an unanswered judge can never stamp pass.** A metric that did not answer is
    `UNMEASURED`, which is a distinct outcome from a metric that answered badly, and
    UNMEASURED never satisfies a tolerance.
  * **audio gates independently of video.** In v1 SNR fell 20.7 → 13.7 dB while SSIM read
    0.85 and the release passed. Here the audio gate is its own verdict with its own
    tolerances, and a green video gate cannot carry it.
  * **no delta is believed before the noise floor is.** Two renders of the same request on
    two hosts differ. Until the protocol carries a measured re-render noise floor for a
    metric, every delta on that metric is reported as UNMEASURED-DELTA and decides nothing.

The metric core is cozy-eval's (ev-001) — adopted, never re-implemented. A second metric
core inside a gate is how two numbers that mean different things end up compared.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NamedTuple

import msgspec
from cozy_runtime.author import (
    App,
    Context,
    FileAsset,
    InvalidRequest,
    Outputs,
    Telemetry,
    Tree,
    UnsupportedInput,
)
from cozy_runtime.derive.identity import canonical_sha256, document_digest, hash_file

app = App()

#: The three verdicts a HUMAN may stamp, and the fourth state that is the absence of one.
VERDICTS = ("free_win", "conditional_parity", "reject")
UNSTAMPED = "unstamped"

#: Verdicts a caller may CLAIM to have stamped. None of them selects anything here
#: (#552.6): a request payload is caller-controlled data, so a stamp arriving on it is a
#: CLAIM this job records and cross-checks against its measurements — never an act of
#: selection authority. Approval is the control plane's authenticated act (unbuilt), and
#: until it exists this job's result says measurements-banked, full stop.
CLAIMS_PASSING = ("free_win",)


class AssetRef(msgspec.Struct, forbid_unknown_fields=True):
    """One rendered asset, bound to the publication that produced it."""

    output_id: str
    digest: str
    """The digest the producing attempt's PublicationReceipt recorded for this output.
    REQUIRED (job-016.2): a digestless asset would be hashed and compared to nothing —
    verification skipped silently — and a verdict about bytes nobody can identify is not
    a verdict."""
    media_type: str
    member: str
    """Its name inside the granted media tree."""

    def __post_init__(self) -> None:
        self.digest = canonical_sha256(self.digest, field="AssetRef.digest")


class ManifestRef(msgspec.Struct, forbid_unknown_fields=True):
    """One side of the pair: what was rendered, by what, under which request."""

    manifest_id: str
    """`cozy.runtime.PublicationReceipt/1`'s own digest — the publication's identity."""
    request_id: str
    artifact: str
    """The exact snapshot digest of the artifact that rendered this."""
    runtime_lane: str
    """Runtime/kernel lane. Two arms differing only here are still two arms."""
    seed: int
    assets: list[AssetRef]
    approximation_site: str = ""
    """WHICH approximation this arm carries — `fp8-gemm`, `fp8-attention`, `curve-adaln`.
    Scored separately against the unapproximated reference: in v1 the fp8 GEMMs were clean
    while fp8 ATTENTION melted detail, and one blended score hid it."""


class Tolerance(msgspec.Struct, forbid_unknown_fields=True):
    metric: str
    lo: float
    hi: float


class DeltaTolerance(msgspec.Struct, forbid_unknown_fields=True):
    """The delta-vs-floor form: |candidate - reference| ≤ k x floor[metric].

    Absolute lo/hi bands derived from a population can be circular about the population
    they judge; this form judges the MOVEMENT against the banked re-render noise floor
    instead, and exists so that fix has a schema slot (job-016.3). It requires a
    `noise_floor` reference, and naming a metric the floor never banked refuses the
    protocol — an unbanked delta cannot be ratified.
    """

    metric: str
    k: float


class StructuralPolicy(msgspec.Struct, forbid_unknown_fields=True):
    """The degenerate-class thresholds — gate POLICY, so they live in the RATIFIED
    protocol (job-016.4). Hard-coded in the gate they could fail a reference without a
    ratified number anywhere. The defaults are the previously hard-coded values."""

    min_motion_energy: float = 0.5
    """Below this the render is FROZEN: no inter-frame motion."""
    noise_hf_ratio: float = 0.25
    noise_shimmer: float = 0.9
    """Together: NOISE_LIKE — high-frequency energy dominates while the detail layer is
    replaced every frame."""


class NoiseFloorRef(msgspec.Struct, forbid_unknown_fields=True):
    """The banked re-render noise floor, as a typed REFERENCE (job-016.3).

    The floor document rides the granted media tree and is bound here by digest; its own
    request/lane binding is cross-checked against the
    protocol and every arm, so borrowing a floor measured under another request or
    runtime lane refuses instead of silently deciding. The floor TABLE is read from the
    verified document, never from the caller."""

    member: str
    """The floor document inside the granted media tree."""
    digest: str
    """The document's sha256 — a floor nobody can identify banks nothing."""

    def __post_init__(self) -> None:
        self.digest = canonical_sha256(self.digest, field="NoiseFloorRef.digest")


class Protocol(msgspec.Struct, forbid_unknown_fields=True):
    """RATIFIED BEFORE any candidate is judged. Its digest is derived, never accepted."""

    reference_inputs: list[str]
    seeds: list[int]
    samples: int
    video: list[Tolerance]
    audio: list[Tolerance]
    conditioning_checks: list[str] = msgspec.field(default_factory=list)
    noise_floor: NoiseFloorRef | None = None
    """The typed floor reference above. Absent means NO delta is believable, and this job
    says so (UNMEASURED-DELTA) rather than reporting one."""
    deltas: list[DeltaTolerance] = msgspec.field(default_factory=list)
    structural: StructuralPolicy = msgspec.field(default_factory=StructuralPolicy)
    ratified_by: str = ""


class VerdictClaim(msgspec.Struct, forbid_unknown_fields=True):
    manifest_id: str
    verdict: str


class GateInput(msgspec.Struct, forbid_unknown_fields=True):
    media: Tree
    protocol: Protocol
    reference: ManifestRef
    candidates: list[ManifestRef]
    verdicts: list[VerdictClaim] = msgspec.field(default_factory=list)
    """candidate `manifest_id` -> the HUMAN verdict. Absent is `unstamped`, which is a
    result and not a default."""


class MetricDelta(msgspec.Struct):
    metric: str
    comparison: str


class ArmResult(msgspec.Struct):
    manifest_id: str
    artifact: str
    runtime_lane: str
    approximation_site: str
    video_gate: str
    audio_gate: str
    video_failed: list[str]
    audio_failed: list[str]
    unmeasured: list[str]
    deltas: list[MetricDelta]
    audio_source: str
    """WHICH asset's soundtrack the audio gate judged, by `output_id`: the dedicated
    audio asset where the arm ships one, else the video asset's embedded track."""
    verdict: str
    """The CALLER-CLAIMED stamp, recorded as data and cross-checked against the measured
    gates — a claim contradicting a measurement refuses. It selects nothing (#552.6)."""


class GateResult(msgspec.Struct):
    evidence: FileAsset
    protocol_id: str
    metric_build: str
    arms: list[ArmResult]
    reference_manifest: str
    passed: int
    rejected: int
    unstamped: int


# ------------------------------------------------------------------ the metric core


def _metric_build() -> str:
    """The exact metric implementation this verdict binds to."""
    import cozy_eval

    return f"cozy-eval/{cozy_eval.__version__}"


def _score(path: Path) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Video statistics, audio statistics, and what did NOT answer.

    A metric that raises is UNMEASURED and is recorded as such. It is never a zero: a
    zero compares, and an unanswered question must not.
    """
    from cozy_eval.metrics.signal import score

    unmeasured: list[str] = []
    clip = score(str(path))
    video = {
        "n_frames": clip.n_frames, "width": clip.width, "height": clip.height,
        "sharpness": clip.sharpness, "sharpness_p10": clip.sharpness_p10,
        "hf_ratio": clip.hf_ratio, "detail_entropy": clip.detail_entropy,
        "contrast": clip.contrast, "saturation": clip.saturation,
        "local_contrast": clip.local_contrast, "brightness": clip.brightness,
        "flicker_pct": clip.flicker, "jerk_ratio": clip.jerk_ratio,
        "shimmer": clip.shimmer, "motion_energy": clip.motion_energy,
        "flat_frames": clip.flat_frames,
    }
    audio, audio_missing = _score_audio(path)
    return video, audio, unmeasured + audio_missing


def _score_audio(path: Path) -> tuple[dict[str, Any], list[str]]:
    """Audio statistics of one file (a soundtrack or a standalone audio asset)."""
    from cozy_eval.audio import read_audio
    from cozy_eval.metrics import audio as audio_metrics

    try:
        stats = audio_metrics.signal_stats(read_audio(str(path)))
        return {k: float(v) for k, v in stats.items()}, []
    except Exception as exc:  # an absent or undecodable soundtrack is a RESULT
        return {}, [f"audio:{type(exc).__name__}"]


def _structural(video: dict[str, Any], policy: StructuralPolicy) -> str:
    """The degenerate classes, stated as facts before any tolerance is consulted.
    Thresholds are the RATIFIED protocol's (job-016.4), never this module's."""
    frames = int(video.get("n_frames") or 0)
    flat = int(video.get("flat_frames") or 0)
    if frames == 0:
        return "DEGENERATE:no frames decoded"
    if flat >= frames:
        return "DEGENERATE:every frame is flat (black, uniform or NaN-filled)"
    if flat:
        return f"PARTIAL_FLAT:{flat} of {frames} frames flat"
    if float(video.get("motion_energy") or 0.0) < policy.min_motion_energy:
        return "FROZEN:no inter-frame motion"
    if (float(video.get("hf_ratio") or 0.0) > policy.noise_hf_ratio
            and float(video.get("shimmer") or 0.0) > policy.noise_shimmer):
        return (
            "NOISE_LIKE:high-frequency energy dominates and the detail layer is "
            "replaced each frame"
        )
    return "COHERENT"


def _gate(values: dict[str, Any], tolerances: list[Tolerance]) -> tuple[list[str], list[str]]:
    failed: list[str] = []
    unmeasured: list[str] = []
    for row in tolerances:
        value = values.get(row.metric)
        if not isinstance(value, (int, float)):
            unmeasured.append(row.metric)
            failed.append(f"{row.metric}=UNMEASURED")
            continue
        if not (row.lo <= float(value) <= row.hi):
            failed.append(f"{row.metric}={float(value):.6g} outside [{row.lo:.6g}, {row.hi:.6g}]")
    return failed, unmeasured


def _deltas(
    candidate: dict[str, Any], reference: dict[str, Any], floor: dict[str, float]
) -> dict[str, str]:
    """Every delta, and whether it is believable. An unbanked noise floor means it is not."""
    out: dict[str, str] = {}
    for metric, value in sorted(candidate.items()):
        base = reference.get(metric)
        if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
            continue
        delta = float(value) - float(base)
        if metric not in floor:
            out[metric] = f"UNMEASURED-DELTA {delta:+.6g} (no re-render noise floor banked)"
        elif abs(delta) <= floor[metric]:
            out[metric] = f"{delta:+.6g} within the {floor[metric]:.6g} noise floor"
        else:
            out[metric] = f"{delta:+.6g} exceeds the {floor[metric]:.6g} noise floor"
    return out


def _resolve(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if not str(candidate).startswith(str(root.resolve()) + os.sep):
        raise UnsupportedInput(f"{relative!r} resolves outside the granted media tree")
    return candidate


def _verify(path: Path, declared: str) -> str:
    got, _ = hash_file(path, window=1 << 20)
    if declared and got != declared:
        raise UnsupportedInput(
            f"{path.name}: the granted bytes are not the published ones — the manifest "
            f"declares {declared} and the tree holds {got}. A verdict about bytes nobody "
            "can identify is not a verdict."
        )
    return got


class _Floor(NamedTuple):
    table: dict[str, float]
    identity: frozenset[str]
    request: str
    lane: str
    digest: str


def _load_floor(root: Path, protocol: Protocol) -> _Floor | None:
    """The banked floor DOCUMENT: located in the granted tree, verified against the
    reference's digest, and read for its own request/lane binding. The table never
    arrives from the caller."""
    ref = protocol.noise_floor
    if ref is None:
        return None
    where = _resolve(root, ref.member)
    if not where.is_file():
        raise UnsupportedInput(
            f"the noise floor document {ref.member!r} is not in the granted media tree"
        )
    _verify(where, ref.digest)
    doc = json.loads(where.read_text())
    expected_fields = {
        "arms", "clip_stats", "hosts", "identity_metrics", "law", "noise_floor",
        "pair_deltas", "request", "runtime_lane", "tuples",
    }
    if not isinstance(doc, dict) or set(doc) != expected_fields:
        actual = sorted(doc) if isinstance(doc, dict) else type(doc).__name__
        raise UnsupportedInput(
            f"{ref.member!r} has fields {actual!r}, expected {sorted(expected_fields)!r}"
        )
    try:
        table = {str(k): float(v) for k, v in doc["noise_floor"].items()}
        request, lane = str(doc["request"]), str(doc["runtime_lane"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UnsupportedInput(f"the floor document is torn: {exc!r}") from exc
    return _Floor(
        table=table,
        identity=frozenset(str(m) for m in doc.get("identity_metrics", [])),
        request=request,
        lane=lane,
        digest=ref.digest,
    )


def _gate_deltas(
    cand_video: dict[str, Any], cand_audio: dict[str, Any],
    ref_video: dict[str, Any], ref_audio: dict[str, Any],
    rows: list[DeltaTolerance], table: dict[str, float],
) -> tuple[list[str], list[str]]:
    """The delta form: |candidate - reference| ≤ k x floor[metric]. UNMEASURED on either
    side never satisfies it."""
    video_failed: list[str] = []
    audio_failed: list[str] = []
    for row in rows:
        in_audio = row.metric in ref_audio or row.metric in cand_audio
        got = (cand_audio if in_audio else cand_video).get(row.metric)
        base = (ref_audio if in_audio else ref_video).get(row.metric)
        sink = audio_failed if in_audio else video_failed
        if not isinstance(got, (int, float)) or not isinstance(base, (int, float)):
            sink.append(f"{row.metric}=UNMEASURED (delta form)")
            continue
        allowed = row.k * table[row.metric]
        delta = float(got) - float(base)
        if abs(delta) > allowed:
            sink.append(
                f"{row.metric} Δ{delta:+.6g} exceeds {row.k:g}xfloor = {allowed:.6g}"
            )
    return video_failed, audio_failed


class _Measured(NamedTuple):
    assets: list[dict[str, Any]]
    """Per-asset evidence rows, keyed by `output_id` (job-016.1)."""
    video: dict[str, Any]
    audio: dict[str, Any]
    unmeasured: list[str]
    structural: str
    audio_source: str


# ------------------------------------------------------------------ the job


@app.job(publishes=True)
def h3_quality_gate(
    ctx: Context,
    payload: GateInput,
    out: Outputs,
    tel: Telemetry,
) -> GateResult:
    """Score every arm against the ratified protocol and bind the result to its identity."""
    root = payload.media.path
    protocol_id = document_digest(
        msgspec.to_builtins(payload.protocol), domain="jobs.quality.protocol"
    )
    build = _metric_build()
    tel.log(f"protocol {protocol_id} · metrics {build}", level="info")

    if not payload.protocol.video and not payload.protocol.audio:
        raise InvalidRequest(
            "the protocol ratifies no tolerance at all: a gate with nothing to fail is not "
            "a gate. Ratify the metrics and their bands before judging a candidate."
        )
    verdicts: dict[str, str] = {}
    for claim in payload.verdicts:
        if claim.manifest_id in verdicts:
            raise InvalidRequest(
                f"duplicate verdict claim for {claim.manifest_id!r} — one candidate has "
                "one human verdict"
            )
        stamped = claim.verdict
        if stamped not in VERDICTS:
            raise InvalidRequest(
                f"{stamped!r} is not a verdict: {' | '.join(VERDICTS)}. This job stamps "
                "none of them by itself — the tri-state verdict is a human act."
            )
        verdicts[claim.manifest_id] = stamped

    # The banked floor, loaded and BOUND before anything is scored (job-016.3): every
    # binding refusal reachable from bounded bytes fires before a frame is decoded.
    floor = _load_floor(root, payload.protocol)
    if payload.protocol.deltas:
        if floor is None:
            raise InvalidRequest(
                "a delta tolerance judges movement against the banked noise floor, and "
                "this protocol banks none — ratify a `noise_floor` reference or drop "
                "the delta form"
            )
        for row in payload.protocol.deltas:
            if row.k <= 0:
                raise InvalidRequest(f"delta tolerance {row.metric}: k={row.k} ratifies nothing")
            if row.metric not in floor.table:
                raise InvalidRequest(
                    f"delta tolerance {row.metric}: the banked floor carries no such "
                    "metric — an unbanked delta cannot be ratified"
                )
    if floor is not None:
        if floor.request not in payload.protocol.reference_inputs:
            raise UnsupportedInput(
                f"the banked floor was measured under request {floor.request!r}, which "
                "this protocol's reference_inputs do not name — a floor borrowed from "
                "another request decides nothing (job-016.3)"
            )
        for bound in (payload.reference, *payload.candidates):
            if bound.runtime_lane != floor.lane:
                raise UnsupportedInput(
                    f"{bound.manifest_id}: rendered on lane {bound.runtime_lane!r} but the "
                    f"banked floor binds lane {floor.lane!r} — a floor borrowed across "
                    "lanes decides nothing (job-016.3)"
                )

    def locate(ref: ManifestRef, asset: AssetRef) -> Path:
        where = _resolve(root, asset.member)
        if not where.is_file():
            raise UnsupportedInput(
                f"{ref.manifest_id}: {asset.member!r} is not in the granted media tree"
            )
        _verify(where, asset.digest)
        return where

    def measure(ref: ManifestRef) -> _Measured:
        """PER-ASSET scoring, keyed by `output_id` (job-016.1): each asset is scored on
        its own, classed video or audio by its declared media type, and nothing
        dict-merges across files. The ratified tolerance vocabulary judges ONE rendered
        clip plus at most one dedicated soundtrack; any wider shape refuses typed rather
        than silently keeping the last asset's verdict."""
        seen: set[str] = set()
        videos: list[AssetRef] = []
        audios: list[AssetRef] = []
        for asset in ref.assets:
            if asset.output_id in seen:
                raise UnsupportedInput(
                    f"{ref.manifest_id}: duplicate output_id {asset.output_id!r} — "
                    "per-asset scores key on it"
                )
            seen.add(asset.output_id)
            if asset.media_type.startswith("video/"):
                videos.append(asset)
            elif asset.media_type.startswith("audio/"):
                audios.append(asset)
            else:
                raise UnsupportedInput(
                    f"{ref.manifest_id}: {asset.output_id!r} is {asset.media_type!r} — "
                    "this gate scores video/* and audio/* assets and refuses what it "
                    "cannot judge"
                )
        if len(videos) != 1:
            raise UnsupportedInput(
                f"{ref.manifest_id}: {len(videos)} video assets — the ratified "
                "tolerances judge exactly ONE rendered clip per arm (with at most one "
                "dedicated audio asset); a wider publication shape needs its own "
                "ratified protocol, not a silent merge (job-016.1)"
            )
        if len(audios) > 1:
            raise UnsupportedInput(
                f"{ref.manifest_id}: {len(audios)} audio assets — one dedicated "
                "soundtrack at most; two cannot share one audio verdict (job-016.1)"
            )

        ctx.raise_if_cancelled()
        v_asset = videos[0]
        v_video, v_audio, v_missing = _score(locate(ref, v_asset))
        structural = _structural(v_video, payload.protocol.structural)
        assets: list[dict[str, Any]] = [{
            "output_id": v_asset.output_id, "kind": "video", "member": v_asset.member,
            "video": v_video, "audio": v_audio, "unmeasured": v_missing,
            "structural": structural,
        }]
        audio_stats = v_audio
        audio_missing = [m for m in v_missing if m.startswith("audio:")]
        audio_source = v_asset.output_id
        unmeasured = [m for m in v_missing if not m.startswith("audio:")]
        if audios:
            ctx.raise_if_cancelled()
            a_asset = audios[0]
            a_stats, a_missing = _score_audio(locate(ref, a_asset))
            assets.append({
                "output_id": a_asset.output_id, "kind": "audio", "member": a_asset.member,
                "video": {}, "audio": a_stats, "unmeasured": a_missing, "structural": "",
            })
            audio_stats, audio_missing, audio_source = a_stats, a_missing, a_asset.output_id
        return _Measured(
            assets=assets, video=v_video, audio=audio_stats,
            unmeasured=unmeasured + audio_missing, structural=structural,
            audio_source=audio_source,
        )

    total_arms = len(payload.candidates) + 1
    with tel.stage("reference", overall_range=(0.0, 1.0 / total_arms)):
        ref_m = measure(payload.reference)
    ref_video, ref_audio = ref_m.video, ref_m.audio
    if ref_m.structural != "COHERENT":
        raise UnsupportedInput(
            f"the REFERENCE arm is {ref_m.structural} — nothing can be gated against it. "
            "A broken reference makes every candidate look fine."
        )

    floor_table = floor.table if floor is not None else {}
    arms: list[ArmResult] = []
    measured_assets: dict[str, list[dict[str, Any]]] = {}
    for index, candidate in enumerate(payload.candidates, start=1):
        with tel.stage(
            f"candidate:{candidate.approximation_site or candidate.manifest_id[:16]}",
            overall_range=(index / total_arms, (index + 1) / total_arms),
        ):
            m = measure(candidate)
        measured_assets[candidate.manifest_id] = m.assets
        video_failed, video_unmeasured = _gate(m.video, payload.protocol.video)
        audio_failed, audio_unmeasured = _gate(m.audio, payload.protocol.audio)
        if m.structural != "COHERENT":
            video_failed.append(m.structural)
        if payload.protocol.deltas and floor is not None:
            delta_video, delta_audio = _gate_deltas(
                m.video, m.audio, ref_video, ref_audio,
                payload.protocol.deltas, floor.table,
            )
            video_failed += delta_video
            audio_failed += delta_audio
        video_gate = "pass" if not video_failed else "fail"
        # The audio gate is decided on its own evidence. A green video gate is not an
        # argument about the soundtrack, and in v1 it was allowed to be one.
        audio_gate = "pass" if not audio_failed else "fail"

        stamped = verdicts.get(candidate.manifest_id, UNSTAMPED)
        if stamped in CLAIMS_PASSING and (video_gate == "fail" or audio_gate == "fail"):
            raise UnsupportedInput(
                f"{candidate.manifest_id}: a {stamped!r} stamp was offered for an arm whose "
                f"measured gate FAILED (video {video_gate}, audio {audio_gate}: "
                f"{'; '.join(video_failed + audio_failed)[:300]}). The stamp is refused, "
                "not recorded — a human verdict overrides a judgement call, never a measurement."
            )
        arms.append(
            ArmResult(
                manifest_id=candidate.manifest_id,
                artifact=candidate.artifact,
                runtime_lane=candidate.runtime_lane,
                approximation_site=candidate.approximation_site,
                video_gate=video_gate,
                audio_gate=audio_gate,
                video_failed=video_failed,
                audio_failed=audio_failed,
                unmeasured=sorted(set(m.unmeasured + video_unmeasured + audio_unmeasured)),
                deltas=[
                    MetricDelta(metric=metric, comparison=comparison)
                    for metric, comparison in sorted({
                        **_deltas(m.video, ref_video, floor_table),
                        **_deltas(m.audio, ref_audio, floor_table),
                    }.items())
                ],
                audio_source=m.audio_source,
                verdict=stamped,
            )
        )
        tel.metric(f"gate.{candidate.manifest_id[:12]}.video_pass", float(video_gate == "pass"))
        tel.metric(f"gate.{candidate.manifest_id[:12]}.audio_pass", float(audio_gate == "pass"))

    document = {
        "protocol_id": protocol_id,
        "protocol": msgspec.to_builtins(payload.protocol),
        "metric_build": build,
        "reference": {
            **msgspec.to_builtins(payload.reference),
            "structural": ref_m.structural,
            "video": ref_video,
            "audio": ref_audio,
            "unmeasured": ref_m.unmeasured,
            "audio_source": ref_m.audio_source,
            "assets": ref_m.assets,
        },
        "arms": [
            {**msgspec.to_builtins(a), "assets": measured_assets[a.manifest_id]}
            for a in arms
        ],
        "noise_floor_banked": sorted(floor_table),
        "noise_floor_binding": (
            {"digest": floor.digest, "request": floor.request, "runtime_lane": floor.lane}
            if floor is not None else None
        ),
        "verdict_policy": {
            "states": [*VERDICTS, UNSTAMPED],
            "note": "measurements-banked is this job's whole authority (#552.6): stamps "
                    "arriving on the payload are recorded claims, cross-checked against "
                    "the measured gates and refused on contradiction — nothing here "
                    "grants serving; that is the control plane's authenticated act, "
                    "which does not exist yet.",
        },
    }
    return GateResult(
        evidence=out.save_bytes(
            json.dumps(document, indent=1, sort_keys=True, default=float).encode(),
            media_type="application/json",
        ),
        protocol_id=protocol_id,
        metric_build=build,
        arms=arms,
        reference_manifest=payload.reference.manifest_id,
        passed=sum(1 for a in arms if a.video_gate == "pass" and a.audio_gate == "pass"),
        rejected=sum(1 for a in arms if a.verdict == "reject"),
        unstamped=sum(1 for a in arms if a.verdict == UNSTAMPED),
    )
