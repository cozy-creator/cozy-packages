#!/usr/bin/env python
"""ev-003's live verification: the real judge, real renders, real scores.

Not a test suite (tracker README #160). Every section drives the `quality-judge` endpoint
through `cozy_runtime.internal.local.run_slice` on the RTX 4070, with cozy-eval's own
prompt builders and parsers on the caller's side, and PRINTS what it observed.

    nice -n 19 .venv/bin/python scripts/judge-live.py smoke        # all four lanes
    nice -n 19 .venv/bin/python scripts/judge-live.py corpus       # judge-v1 adherence
    nice -n 19 .venv/bin/python scripts/judge-live.py separation   # owner-labelled arms
    nice -n 19 .venv/bin/python scripts/judge-live.py arms         # the red arms
    nice -n 19 .venv/bin/python scripts/judge-live.py bench

THE QUESTION THIS ANSWERS. A judge that cannot tell a defective clip from a clean one is a
finding, not a feature, so the sections are built around discrimination the corpus already
has ground truth for:

  corpus      each clip's OWN authored checklist against each clip's frames, and every
              clip's checklist against a DIFFERENT clip's frames. The mismatched cell is
              ground truth for "no": the items are about things visibly not there.
  separation  ev-005's owner-labelled populations. `h3-freesel-mn-k16` is 8 pairs of an
              owner-REJECTED sparse-attention arm against its dense reference;
              `h3-freesel-seed-control` is 4 pairs of the SAME arm at two seeds, which a
              judge must NOT separate. A blinded pairwise read of both, with the endpoint's
              arm-order swap, is the judge's own null control.

Media comes from `~/cozy/samples` (v1, READ-ONLY) through the corpus's digest-verified
blob resolution — nothing is copied and nothing is written there.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "quality-judge"))

from cozy_eval import corpus, detail, frames, video, wire  # noqa: E402
from cozy_eval.errors import BackendError  # noqa: E402
from cozy_eval.metrics import temporal  # noqa: E402
from cozy_eval.metrics.adherence import ask_judge  # noqa: E402

import quality_judge as qj  # noqa: E402  — the endpoint's OWN admission bounds
from slice import SliceTransport  # noqa: E402

EVAL = Path.home() / "cozy_v2" / "cozy-eval"
CORPUS = EVAL / "calibration" / "corpus" / "judge-v1"
SAMPLES = Path.home() / "cozy" / "samples"
FREESEL = "h3-freesel-20260811/out/idx"

#: Frames per judged strip. Every frame costs ~392 visual tokens on this binding, and a
#: pairwise call shows TWO strips, so four is what keeps an 8 GiB card's context honest.
STRIP = 4

#: The owner-labelled populations, from ev-005's `calibration/populations.json`.
CELLS = ("busker", "carpet", "glassblower", "whale")
SEEDS = ("s20260808", "s20260811")

PASS = 0
FAIL = 0


def head(title: str) -> None:
    print(f"\n=== {title}", flush=True)


def check(label: str, ok: bool, detail_text: str = "") -> bool:
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f" — {detail_text}" if detail_text else ""),
          flush=True)
    return ok


# --------------------------------------------------------------------------- media


def strip_of(path: Path, count: int = STRIP) -> list[Any]:
    """An ordered strip of PIL frames — the same construction the video lane uses.

    STREAMED. Materializing a 10 s 1344x768 clip as float32 is 2.2 GiB for four frames
    that survive, and it is what made the first run of this script unusable — measured, so
    the fix is here rather than a note. `frames.iter_video` is O(1) RAM by design; the
    sampler keeps only the indices `temporal.sample_indices` names.
    """
    import numpy as np

    _, _, _, declared = frames.probe(path)
    wanted: set[int] | None = (
        set(temporal.sample_indices(declared, count)) if declared > 0 else None
    )
    kept: list[Any] = []
    total = 0
    for index, frame in enumerate(frames.iter_video(path)):
        total += 1
        if wanted is None or index in wanted:
            kept.append(frame)
    if wanted is not None and total == declared:
        images: list[Any] = temporal.frame_images(np.stack(kept))
        return images
    # The container lied about its frame count: fall back to the frames we kept, sampled
    # again over what actually arrived, rather than returning a strip that is not uniform.
    pool = np.stack(kept)
    sampled: list[Any] = temporal.frame_images(pool[temporal.sample_indices(len(pool), count)])
    return sampled


def degrade(strip: list[Any], kind: str) -> list[Any]:
    """The three defect classes ev-001 calibrates its integrity floor against."""
    import numpy as np
    from PIL import Image, ImageFilter

    out = []
    for image in strip:
        if kind == "blur":
            out.append(image.filter(ImageFilter.GaussianBlur(radius=6)))
        elif kind == "noise":
            array = np.asarray(image).astype(np.float32)
            noise = np.random.default_rng(7).normal(0, 64, array.shape)
            out.append(Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8)))
        elif kind == "blank":
            out.append(Image.new("RGB", image.size, (16, 16, 16)))
        else:
            raise ValueError(kind)
    return out


# --------------------------------------------------------------------------- sections


@dataclass
class Judges:
    transport: SliceTransport
    judge: wire.WireJudge = field(init=False)
    soft: wire.WireSoftJudge = field(init=False)
    pair: wire.WirePairwise = field(init=False)
    asr: wire.WireTranscriber = field(init=False)

    def __post_init__(self) -> None:
        # The endpoint's admission bounds come from the endpoint MODULE, not from a copy
        # in the client: one strip per call for the judge lanes, two for a comparison.
        single = min(qj.MAX_CALLS, qj.MAX_IMAGES // STRIP)
        paired = min(qj.MAX_CALLS, qj.MAX_IMAGES // (2 * STRIP))
        self.judge = wire.WireJudge(self.transport, max_calls=single)
        self.soft = wire.WireSoftJudge(self.transport, max_calls=single)
        self.pair = wire.WirePairwise(self.transport, max_calls=paired)
        self.asr = wire.WireTranscriber(self.transport, max_calls=qj.MAX_CALLS)


def section_smoke(j: Judges) -> None:
    """All four lanes, on ONE clip, in the fewest requests that can prove they run."""
    head("smoke — the four protocol lanes on one real render")
    loaded = corpus.load(CORPUS)
    row = loaded.rows[0]
    clip = loaded.blob(row.exemplar, snapshot_root=str(SAMPLES))
    strip = strip_of(clip)

    items = row.items
    prompt = video.build_video_judge_prompt(items, frame_count=len(strip))
    j.judge.prefetch([(strip, prompt)])
    answers, reason = ask_judge(j.judge, strip, prompt, len(items))
    check(f"judge answered {row.id}", not reason and len(answers) == len(items),
          reason or f"{len(answers)}/{len(items)} answers: "
                    f"{[('yes' if answers[i] else 'no') for i in sorted(answers)]}")

    question = items[0].question
    p = j.soft.p_yes(strip, question)
    check("soft p(yes) is a probability", 0.0 <= p <= 1.0, f"p={p:.4f} for {question!r}")

    reads = j.pair.compare([("self", strip, strip, detail._PAIRWISE_PREAMBLE.format(n=len(strip)))])
    check("pairwise returned both orderings", "self" in reads and all(reads["self"]),
          f"forward={reads.get('self', ('', ''))[0]!r} swapped={reads.get('self', ('', ''))[1]!r}")

    samples, rate = _audio(clip)
    text = j.asr.transcribe(samples, rate)
    check("transcriber answered", isinstance(text, str),
          f"{len(samples) / rate:.1f}s -> {text[:110]!r}")


def _audio(path: Path) -> tuple[Any, int]:
    from cozy_eval.audio import read_audio

    clip = read_audio(path)
    return clip.samples, clip.sample_rate


def section_corpus(j: Judges) -> None:
    """The authored checklist, MATCHED and MISMATCHED, plus the defect ladder."""
    head("corpus — judge-v1 adherence on nine real H3 renders")
    loaded = corpus.load(CORPUS)
    rows = list(loaded.rows)
    strips = {
        row.id: strip_of(loaded.blob(row.exemplar, snapshot_root=str(SAMPLES))) for row in rows
    }
    print(f"  corpus {loaded.name} rows_digest {loaded.rows_digest}", flush=True)

    # ONE request carries every call: matched, mismatched (the next row's frames), and the
    # three defect ladders on the matched pair.
    work: list[tuple[str, str, list[Any], tuple[Any, ...]]] = []
    for i, row in enumerate(rows):
        other = rows[(i + 1) % len(rows)]
        prompt = video.build_video_judge_prompt(row.items, frame_count=STRIP)
        work.append((f"{row.id}|matched", prompt, strips[row.id], row.items))
        work.append((f"{row.id}|mismatched:{other.id}", prompt, strips[other.id], row.items))
        for kind in ("blur", "noise", "blank"):
            work.append((f"{row.id}|{kind}", prompt, degrade(strips[row.id], kind), row.items))

    started = time.perf_counter()
    j.judge.prefetch([(strip, prompt) for _, prompt, strip, _ in work])
    took = time.perf_counter() - started
    print(f"  {len(work)} judge calls in {j.judge.requests} request(s), {took:.1f}s "
          f"({took / len(work):.2f}s/call)", flush=True)

    recalls: dict[str, list[float]] = {}
    per_row: dict[str, dict[str, float]] = {}
    unmeasured = 0
    for label, prompt, strip, items in work:
        row_id, _, condition = label.partition("|")
        answers, reason = ask_judge(j.judge, strip, prompt, len(items))
        if reason:
            unmeasured += 1
            continue
        yes = sum(1 for i in range(1, len(items) + 1) if answers.get(i))
        recall = yes / len(items)
        recalls.setdefault(condition.split(":")[0], []).append(recall)
        per_row.setdefault(row_id, {})[condition.split(":")[0]] = recall

    print("\n  element_recall by condition (weighted fraction of authored items verified)")
    print(f"  {'row':<26}" + "".join(f"{c:>12}" for c in ("matched", "mismatched", "blur",
                                                          "noise", "blank")))
    for row in rows:
        cells = per_row.get(row.id, {})
        print(f"  {row.id:<26}" + "".join(
            f"{cells.get(c, float('nan')):>12.3f}"
            for c in ("matched", "mismatched", "blur", "noise", "blank")))
    print(f"  {'MEAN':<26}" + "".join(
        f"{statistics.fmean(recalls.get(c, [0.0])):>12.3f}"
        for c in ("matched", "mismatched", "blur", "noise", "blank")))

    matched = statistics.fmean(recalls["matched"])
    mismatched = statistics.fmean(recalls["mismatched"])
    check("matched scores above mismatched", matched > mismatched,
          f"{matched:.3f} vs {mismatched:.3f} (separation {matched - mismatched:+.3f})")
    check("a blank frame scores below its own clip",
          statistics.fmean(recalls["blank"]) < matched,
          f"blank {statistics.fmean(recalls['blank']):.3f} vs matched {matched:.3f}")
    check("no call was lost", unmeasured == 0, f"{unmeasured} unmeasured")


def section_detail(j: Judges) -> None:
    """detail.DETAIL_AXES over each clip clean and degraded — the fine-detail rubric."""
    head("detail — the fine-detail rubric (cozy_eval.detail) clean vs degraded")
    loaded = corpus.load(CORPUS)
    rows = list(loaded.rows)
    strips = {
        row.id: strip_of(loaded.blob(row.exemplar, snapshot_root=str(SAMPLES))) for row in rows
    }
    conditions = ("clean", "blur", "noise")
    # Degrade ONCE per (row, condition). A `degrade()` inside a comprehension that also
    # loops the four axes ran a radius-6 Gaussian over 4 x 1 344 x 768 seventy-two times
    # for seventy-two identical results, and it was most of this section's wall clock.
    ladder = {
        (row.id, condition): (
            strips[row.id] if condition == "clean" else degrade(strips[row.id], condition)
        )
        for row in rows
        for condition in conditions
    }
    work = [(f"{row.id}|{c}", ladder[(row.id, c)]) for row in rows for c in conditions]
    prompt = detail.build_detail_judge_prompt(frame_count=STRIP)
    j.judge.prefetch([(strip, prompt) for _, strip in work])

    scores: dict[str, list[float]] = {c: [] for c in conditions}
    axes: dict[str, dict[str, list[float]]] = {c: {} for c in conditions}
    for label, strip in work:
        _, _, condition = label.partition("|")
        read = detail.score_detail_vlm(strip, j.judge)
        if not read:
            continue
        scores[condition].append(statistics.fmean(read.values()))
        for axis, value in read.items():
            axes[condition].setdefault(axis, []).append(value)

    print(f"  {'axis':<26}" + "".join(f"{c:>10}" for c in conditions))
    for axis, _ in detail.DETAIL_AXES:
        print(f"  {axis:<26}" + "".join(
            f"{statistics.fmean(axes[c].get(axis, [float('nan')])):>10.3f}" for c in conditions))
    print(f"  {'MEAN (clean=1.0 is good)':<26}" + "".join(
        f"{statistics.fmean(scores[c] or [float('nan')]):>10.3f}" for c in conditions))
    check("the rubric scores clean above blurred",
          statistics.fmean(scores["clean"]) > statistics.fmean(scores["blur"]),
          f"{statistics.fmean(scores['clean']):.3f} vs {statistics.fmean(scores['blur']):.3f}")
    check("the rubric scores clean above noised",
          statistics.fmean(scores["clean"]) > statistics.fmean(scores["noise"]),
          f"{statistics.fmean(scores['clean']):.3f} vs {statistics.fmean(scores['noise']):.3f}")

    # The HARD parse can only say yes or no. When it saturates, the question a report has
    # to answer is whether the model is blind or the READ is coarse — which is exactly what
    # the soft lane exists for (`vqascore`: "soft scores resolve small deltas a hard parse
    # rounds away"). Same axes, same frames, p(yes) instead of a token.
    head("detail — the same axes read SOFT: p(yes) instead of a parsed token")
    soft_work = [
        (f"{row.id}|{condition}", axis, question, ladder[(row.id, condition)])
        for row in rows
        for condition in conditions
        for axis, question in detail.DETAIL_AXES
    ]
    started = time.perf_counter()
    j.soft.prefetch([(strip, question) for _, _, question, strip in soft_work])
    print(f"  {len(soft_work)} soft calls in {j.soft.requests} request(s), "
          f"{time.perf_counter() - started:.1f}s wall, "
          f"{j.soft.infer_seconds / len(soft_work):.2f}s/call inference", flush=True)
    soft: dict[str, dict[str, list[float]]] = {c: {} for c in conditions}
    for label, axis, question, strip in soft_work:
        _, _, condition = label.partition("|")
        soft[condition].setdefault(axis, []).append(j.soft.p_yes(strip, question))
    print(f"  {'axis':<26}" + "".join(f"{c:>10}" for c in conditions) + "   clean-blur")
    for axis, _ in detail.DETAIL_AXES:
        means = [statistics.fmean(soft[c][axis]) for c in conditions]
        print(f"  {axis:<26}" + "".join(f"{m:>10.4f}" for m in means)
              + f"{means[0] - means[1]:>13.4f}")
    overall = [statistics.fmean([v for a in soft[c].values() for v in a]) for c in conditions]
    print(f"  {'MEAN p(yes)':<26}" + "".join(f"{m:>10.4f}" for m in overall)
          + f"{overall[0] - overall[1]:>13.4f}")
    check("the SOFT read separates clean from blurred where the hard parse saturates",
          overall[0] > overall[1], f"{overall[0]:.4f} vs {overall[1]:.4f} "
          f"(separation {overall[0] - overall[1]:+.4f})")
    check("the SOFT read separates clean from noised",
          overall[0] > overall[2], f"{overall[0]:.4f} vs {overall[2]:.4f} "
          f"(separation {overall[0] - overall[2]:+.4f})")


def _freesel(cell: str, arm: str, seed: str) -> Path:
    return SAMPLES / FREESEL / f"{cell}-5s__{arm}-{seed}.mp4"


def section_separation(j: Judges) -> None:
    """The judge's OWN red arm, on ev-005's owner-labelled populations."""
    head("separation — owner-REJECTED mn-k16 vs its dense reference, and the null control")
    gate = [(f"{cell}/{seed}", _freesel(cell, "dense", seed), _freesel(cell, "mn-k16", seed))
            for cell in CELLS for seed in SEEDS]
    control = [
        (f"{cell}/null", _freesel(cell, "dense", SEEDS[0]), _freesel(cell, "dense", SEEDS[1]))
        for cell in CELLS
    ]
    missing = [str(p) for _, a, b in gate + control for p in (a, b) if not p.is_file()]
    if missing:
        raise SystemExit(f"missing banked renders: {missing[:4]}")

    strips = {}
    for label, ref, cand in gate + control:
        strips[label] = (strip_of(ref), strip_of(cand))
    prompt = detail._PAIRWISE_PREAMBLE.format(n=STRIP)

    # A = the REFERENCE arm in every call, so a position-biased judge shows up as a
    # systematic lean rather than as a result. The endpoint swaps the arms itself.
    work = [(label, ref, cand, prompt) for label, (ref, cand) in strips.items()]
    started = time.perf_counter()
    reads = j.pair.compare(work)
    took = time.perf_counter() - started
    print(f"  {len(work)} blinded comparisons x 2 orderings in {j.pair.requests} "
          f"request(s), {took:.1f}s", flush=True)

    print(f"\n  {'pair':<20}{'forward':>10}{'swapped':>10}{'verdict':>12}  population")
    tally: dict[str, dict[str, int]] = {"gate": {}, "control": {}}
    flips = 0
    for label, _ref, _cand, _prompt in work:
        population = "control" if label.endswith("/null") else "gate"
        forward, swapped = reads.get(label, ("", ""))
        a = _winner(forward)
        b = _winner(swapped)
        # forward shows (reference, candidate) so "A" is the reference; swapped shows
        # (candidate, reference) so "A" is the candidate. A judge that answers the CONTENT
        # gives the same arm both times; one that answers the POSITION gives "A" twice.
        verdict = _agree(a, b)
        if a == b and a in ("A", "B"):
            flips += 1
        tally[population][verdict] = tally[population].get(verdict, 0) + 1
        print(f"  {label:<20}{a:>10}{b:>10}{verdict:>12}  {population}")

    decided_keys = ("reference", "candidate")
    print(f"\n  gate    (owner REJECTED the candidate): {tally['gate']}")
    print(f"  control (identical weights, seeds only):  {tally['control']}")
    print(f"  position-bias: {flips}/{len(work)} pairs named the same POSITION both times")
    # THE CONTROL IS THE POINT. Without the swap, the naive read of this population is
    # "B wins 9 of 12" and someone ships a verdict; with it, most of those reads are the
    # judge naming a SLOT. So the arm that must fire is "the swap detects the bias", and
    # the arm that must then be reported honestly is whether anything survives it.
    check("the arm-order swap detects position bias", flips > 0,
          f"{flips}/{len(work)} pairs named the same POSITION in both orderings")
    stable = sum(v for k, v in tally["gate"].items() if k in decided_keys)
    check("a majority of the owner-rejected pairs yield a content-stable verdict",
          stable * 2 > len(gate),
          f"{stable}/{len(gate)} stable "
          f"(reference {tally['gate'].get('reference', 0)}, "
          f"candidate {tally['gate'].get('candidate', 0)}); "
          "a FAIL here is the judge's own red arm — this checkpoint cannot gate this arm")
    control_stable = sum(v for k, v in tally["control"].items() if k in decided_keys)
    check("the null control does not separate MORE than the gate population",
          control_stable / len(control) <= max(stable / len(gate), 1e-9) + 1e-9,
          f"control {control_stable}/{len(control)} stable vs gate {stable}/{len(gate)}")


def _winner(raw: str) -> str:
    import re

    match = re.search(r"\{[^{}]*\}", raw or "")
    if not match:
        return "?"
    try:
        winner = str(json.loads(match.group(0)).get("winner", "")).strip().lower()
    except (ValueError, TypeError):
        return "?"
    return {"a": "A", "b": "B", "tie": "tie"}.get(winner, "?")


def _agree(forward: str, swapped: str) -> str:
    """The arm both orderings name, or why there is no verdict."""
    if forward == "?" or swapped == "?":
        return "unparsed"
    if forward == "tie" or swapped == "tie":
        return "tie"
    if forward == swapped:  # the same POSITION won twice: the judge read the slot
        return "position"
    return "reference" if forward == "A" else "candidate"


def section_arms(j: Judges) -> None:
    """Every red arm ev-003 names, observed firing."""
    head("arms — the red arms")
    loaded = corpus.load(CORPUS)
    row = loaded.rows[0]
    strip = strip_of(loaded.blob(row.exemplar, snapshot_root=str(SAMPLES)))

    # 1. An undecodable image is a typed REQUEST refusal, not a backend fault.
    junk = j.transport.stage(b"not-an-image", "image/jpeg")
    try:
        j.transport.invoke("judge", {"calls": [
            {"id": "x", "prompt": "1. anything?", "images": [{"image": junk}]}]})
    except Exception as exc:
        check("an undecodable image refuses the attempt whole",
              "not decodable" in str(exc) and "REFUSED" in str(exc),
              str(exc).splitlines()[0][:170])
    else:
        check("an undecodable image refuses the attempt", False, "it succeeded")

    # 2. A judge that FAILS leaves items unmeasured — never a "no".
    class Failing:
        model_ref = "stub://always-fails"

        def ask(self, images: list[Any], prompt: str) -> str:
            raise BackendError("the endpoint refused this call")

    answers, reason = ask_judge(Failing(), strip, "1. anything?", 1)
    check("a failing judge yields a reason and no answers", not answers and bool(reason), reason)
    score = video.score_video(loaded.checklist_set().t2v[row.id], strip, judge=Failing())
    check("its items land UNMEASURED, never verified-false",
          not score.measured and "UNMEASURED" in score.note, score.note[:150])

    # 3. An unparseable reply is the same class as a failure.
    class Babbling:
        model_ref = "stub://unparseable"

        def ask(self, images: list[Any], prompt: str) -> str:
            return "I think the video looks quite nice overall."

    answers, reason = ask_judge(Babbling(), strip, "1. anything?", 1)
    check("an unparseable reply yields a reason and no answers", not answers and bool(reason),
          reason)

    # 4. A judge that answers only SOME questions: the answered ones score, the rest are
    #    excluded from the denominator and named.
    class Partial:
        model_ref = "stub://partial"

        def ask(self, images: list[Any], prompt: str) -> str:
            return '[{"n": 1, "answer": "yes"}]'

    partial = video.score_video(loaded.checklist_set().t2v[row.id], strip, judge=Partial())
    check("a partial reply scores what was answered and names the rest",
          partial.measured and len(partial.items) == 1 and "UNMEASURED" in partial.note,
          f"recall {partial.element_recall:.3f} over {len(partial.items)} item(s); "
          f"{partial.note[:110]}")

    # 5. The audio-event tier REFUSES rather than inventing a score.
    audio_judge = wire.WireAudioJudge(j.transport)
    try:
        audio_judge.ask(None, 48_000, "1. is there a drum hit?")
    except BackendError as exc:
        check("the audio-event tier refuses typed", "no audio-event judge" in str(exc),
              str(exc)[:120])
    else:
        check("the audio-event tier refuses typed", False, "it answered")

    # 6. A checklist item the sample VISIBLY LACKS scores no.
    absent = video.build_video_judge_prompt(
        tuple(i for i in loaded.rows[7].items), frame_count=STRIP
    )
    j.judge.prefetch([(strip, absent)])
    answers, reason = ask_judge(j.judge, strip, absent, len(loaded.rows[7].items))
    yes = sum(1 for v in answers.values() if v)
    check("a foreign checklist scores mostly no on this clip", yes <= 1,
          f"{yes}/{len(answers)} yes ({loaded.rows[7].id} items on {row.id} frames)")


def section_bench(j: Judges) -> None:
    """Latency, batch amortization, VRAM."""
    head("bench")
    loaded = corpus.load(CORPUS)
    rows = list(loaded.rows)
    strips = [strip_of(loaded.blob(r.exemplar, snapshot_root=str(SAMPLES))) for r in rows]
    prompt = video.build_video_judge_prompt(rows[0].items, frame_count=STRIP)

    print(f"  {'batch':>8}{'wall s':>10}{'ready s':>10}{'attempt s':>12}"
          f"{'s/call':>10}{'peak MiB':>11}")
    for batch in (1, 4, 9):
        client = wire.WireJudge(j.transport)
        started = time.perf_counter()
        client.prefetch([(strips[i % len(strips)], prompt) for i in range(batch)])
        wall = time.perf_counter() - started
        run = j.transport.runs[-1]
        timings = run.get("timings", {})
        metrics = run.get("metrics", {})
        print(f"  {batch:>8}{wall:>10.1f}"
              f"{timings.get('register_to_ready_ms', 0) / 1000:>10.1f}"
              f"{timings.get('request_to_result_ms', 0) / 1000:>12.1f}"
              f"{client.infer_seconds / batch:>10.2f}"
              f"{float(metrics.get('peak_vram_bytes', 0)) / (1 << 20):>11.0f}")

    samples, rate = _audio(loaded.blob(rows[0].exemplar, snapshot_root=str(SAMPLES)))
    asr = wire.WireTranscriber(j.transport)
    started = time.perf_counter()
    asr.prefetch([(samples, rate)])
    wall = time.perf_counter() - started
    run = j.transport.runs[-1]
    print(f"\n  transcribe {len(samples) / rate:.1f}s of audio: {wall:.1f}s wall, "
          f"{asr.infer_seconds:.2f}s infer, ready "
          f"{run.get('timings', {}).get('register_to_ready_ms', 0) / 1000:.1f}s, peak "
          f"{float(run.get('metrics', {}).get('peak_vram_bytes', 0)) / (1 << 20):.0f} MiB")


SECTIONS = {
    "smoke": section_smoke,
    "corpus": section_corpus,
    "detail": section_detail,
    "separation": section_separation,
    "arms": section_arms,
    "bench": section_bench,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sections", nargs="*", default=[], choices=[*SECTIONS, []])
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    wanted = args.sections or list(SECTIONS)

    transport = SliceTransport()
    judges = Judges(transport)
    started = time.perf_counter()
    for name in wanted:
        SECTIONS[name](judges)
    print(f"\n{PASS} ok, {FAIL} FAIL — {time.perf_counter() - started:.1f}s, "
          f"{len(transport.runs)} endpoint requests", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(transport.runs, indent=1))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
