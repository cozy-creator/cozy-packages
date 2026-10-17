"""The exact text H3's encoder receives, built by the package's own prompt code; no model.

One-off calls send their prompt verbatim, plus an `N/A` music line when it names none.
long_form segments also gain the global sections, `style` among them, and only the
references they name. Unknown request fields are ignored; a missing required one refuses.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import msgspec
from msgspec.structs import replace
from cozy_runtime.author import Assets, AudioAsset, ImageAsset, InvalidRequest, VideoAsset
from cozy_runtime.author.fakes import fake_attempt, fake_context, fake_outputs, fake_telemetry
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import h3
from minimax_h3.official import Task
from story import (
    SHORT_DESCRIPTION,
    StoryReference,
    apply_written,
    compile_segments,
    encoder_prompt,
    image_prompt,
    opening_cut_warnings,
    resolve_reference_images,
    validate_references,
    writer_request,
)

# The owner's own ref2va prompt, verbatim from MiniMax's full-reference guide.
COFFEE = """subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, <Picture 3>, and <Picture 4>, with thick white fur, pointed ears, a dark nose, and a curved tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with rolled-up sleeves.
<Subject 4> is the young man in <Video 2>, with short wavy brown hair and a dark-grey hoodie with drawstrings.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. <Subject 4> enters with <Subject 2>, which lunges toward the cookie. The three-shot exchange uses <Audio 1> as the voice-timbre reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose, and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the blonde woman's identity, long hair, and light-pink shirt are retained.
<Subject 4> (appears in [Shot 1], [Shot 2]): fully_preserved - the young man's short wavy brown hair and dark-grey hoodie are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa, patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, <Subject 4>, the young man with short wavy brown hair and a dark-grey hoodie with drawstrings, enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance, <d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while <Subject 4> pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 4> (S2), the young man in the dark-grey hoodie from Shot 1, sitting beside <Subject 3> on the sofa and holding <Subject 2> securely in his arms. <Subject 4> (S2) says in a casual young male voice with a playful tone and an easy conversational pace, <d>[English] He just likes cookies more than me.</d> He closes his mouth into an apologetic smile and strokes the dog's thick white fur.
[Shot 3] At 00:05.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from Shot 1. Her annoyance softens as she looks toward the Samoyed. <Subject 3> (S1) replies in the same clear youthful voice referenced from <Audio 1> with an amused cadence, <d>[English] Well, he has good taste at least.</d> She smiles and raises the cookie in a small toast-like gesture. A classic canned audience laugh begins immediately after the line and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene.

non_diegetic_music:
N/A"""
# The owner's run-1559 segment, pasted as free text.
CHIP = """summary:
[reference generation] Finally <Subject-4> pulls a computer chip out from <Subject-2>'s inner-pocket of her shirt.

detailed_description:
[Shot 1] <Subject-4> searches <Subject-2>'s body thoroughly, and finally finds a computer chip. She picks it up and puts it in her pocket, and walks away. The camera pans on <Subject-2> <Subject-3>'s bodies laying on the ground.

overall_soundscape:
Environmental ambience continues across segments.

non_diegetic_music:
N/A"""
ONE_LINE = "A red kite climbs over an empty beach."
SCENE = "the architecture, materials and defining features of the referenced environment"
PERSON = (
    "keep the face, hair, and outfit of the referenced character. Ignore the white background"
)
LAB = [
    {"name": "Subject-1", "kind": "character", "description": "A tall courier in a red jacket."},
    {"name": "Subject-2", "kind": "character", "description": "An adult woman in a black tactical suit."},
    {"name": "Subject-3", "kind": "character", "description": "An adult man in a grey security uniform."},
    {"name": "Subject-4", "kind": "character", "description": "An adult woman in a white lab coat."},
    {"name": "Background", "kind": "background", "description": "An underground laboratory with steel benches."},
]
HUM = "A low hum of laboratory machinery fills the hall."


def request(job: Any, wire: Any) -> Any:
    """Decode the wire as Runtime does; an asset handle stands in for granted bytes."""

    def handle(kind: type, value: Any) -> Any:
        if kind in (AudioAsset, ImageAsset, VideoAsset):
            return kind(value)
        raise NotImplementedError(kind)

    return msgspec.convert(wire, type=job, dec_hook=handle)


def check(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise AssertionError(f"{name}\n--- actual ---\n{actual}\n--- expected ---\n{expected}")
    print(f"  ok   {name}")


def one_offs() -> None:
    tasks: tuple[Task, ...] = ("ref2va", "ref2va_turbo", "fl2va", "fl2va_turbo")
    for task in tasks:
        wire = get_type_hints(getattr(h3, task))["payload"]
        music = "non_diegetic_music: N/A" if task.startswith("fl2va") else "non_diegetic_music:\nN/A"
        for label, prompt, sent in (
            ("the owner's coffee-shop prompt", COFFEE, COFFEE),
            ("the owner's chip segment as free text", CHIP, CHIP),
            ("a one-line prompt", ONE_LINE, f"{ONE_LINE}\n\n{music}"),
            ("trailing whitespace", ONE_LINE + "\n\n", f"{ONE_LINE}\n\n{music}"),
            ("music named in any case", "Kites.\nNON_DIEGETIC_MUSIC: harp", "Kites.\nNON_DIEGETIC_MUSIC: harp"),
            ("music heading with spaces", "Kites.\nNon Diegetic Music: harp", "Kites.\nNon Diegetic Music: harp"),
            ("music heading with hyphens", "Kites.\nnon-diegetic-music: harp", "Kites.\nnon-diegetic-music: harp"),
            ("an empty prompt", "", music),
        ):
            payload = request(wire, {"prompt": prompt, "duration_s": 15})
            check(f"{task}: {label}", encoder_prompt(task, payload.prompt), sent)


def calls(job: Any, wire: dict[str, Any]) -> list[tuple[str, list[str], list[str]]]:
    """What each child call receives: its encoder text, attached references and warnings."""
    child: Task = "ref2va_turbo" if job is h3.LongFormInput else "ref2va"
    return [
        (encoder_prompt(child, call.prompt), [ref.name for ref in call.references], call.warnings)
        for call in h3.segment_calls(request(job, wire))
    ]


def music() -> None:
    score = "A gentle string quartet, no vocals."
    scene = {"name": "Beach", "kind": "background", "description": "An empty beach."}
    prefix = (
        "subject_definitions:\n<Beach> is the background shown in <Picture 1>. An empty beach.\n\n"
        f"retention_analysis:\n<Beach>: fully_preserved - {SCENE}.\n\n"
        f"detailed_description:\n{ONE_LINE}\n\n"
    )

    class TokenizerInput:
        """Observe the actual upstream presentation's tokenizer input without model weights."""

        def __init__(self) -> None:
            self.text: list[str] = []

        def __call__(self, value: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            assert not add_special_tokens
            self.text.append(value)
            return {"input_ids": list(value.encode())}

    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        wire: dict[str, Any] = {
            "references": [scene], "non_diegetic_music": score,
            "segments": [{"prompt": ONE_LINE, "duration_s": 5} for _ in range(5)],
        }
        expected = [prefix + f"non_diegetic_music:\n{score}"] * 5
        check(f"{job.__name__}: global score reaches all five segments",
              [row[0] for row in calls(job, wire)], expected)
        for segment, value in zip(wire["segments"][1:], (None, "Solo piano.", "", "N/A"), strict=True):
            segment["non_diegetic_music"] = value
        expected = [prefix + f"non_diegetic_music:\n{value}"
                    for value in (score, score, "Solo piano.", "N/A", "N/A")]
        actual = [row[0] for row in calls(job, wire)]
        check(f"{job.__name__}: omitted/null inherit and explicit music replaces", actual, expected)
        for prompt in actual:
            observed = TokenizerInput()
            tokens, _ = MiniMaxH3Ref2VATextEncoderStep._build_presentation(
                observed, prompt, [], [], [], [],
            )
            check(f"{job.__name__}: upstream tokenizer receives the exact composed score",
                  (observed.text, tokens), ([prompt], list(prompt.encode())))
        inline = ONE_LINE + "\n\nnon_diegetic_music:\nInline harp."
        for override, wanted in ((None, "Inline harp."), ("Drums.", "Drums."), ("N/A", "N/A")):
            wire["segments"] = [{"prompt": inline, "duration_s": 5, "non_diegetic_music": override}]
            check(f"{job.__name__}: inline score with override {override!r}",
                  calls(job, wire)[0][0], prefix + f"non_diegetic_music:\n{wanted}")
        referenced = {
            "references": [scene, *[
                {"name": name, "kind": "audio", "audio": "sha256:" + digit * 64}
                for name, digit in (("OldScore", "a"), ("ChosenScore", "b"))
            ]],
            "segments": [{
                "prompt": ONE_LINE + "\n\nnon_diegetic_music:\nUse <Audio 1>.",
                "duration_s": 5,
                "non_diegetic_music": "Play <Audio 2> softly.",
            }],
        }
        text, attached, warnings = calls(job, referenced)[0]
        check(f"{job.__name__}: only the overriding score is attached",
              (attached, warnings), (["Beach", "ChosenScore"], []))
        check(f"{job.__name__}: overriding audio is renumbered to its child slot",
              text.endswith("non_diegetic_music:\nPlay <Audio 1> softly."), True)
        check(f"{job.__name__}: overwritten score is absent", "OldScore" in text, False)
    check("music override stays out of one-off ClipInput", "non_diegetic_music" in h3.ClipInput.__struct_fields__, False)


def long_form() -> None:
    segments = [
        # The owner's example: all four sections are added around summary and description.
        "summary:\n[reference generation] <Subject-4> explores.\n\ndetailed_description:\n"
        "[Shot 1] <Subject-4> is walking through the labratory <Background>, exploring and "
        "looking around.",
        # Sections already present gain the global content inside them.
        "Subject Definitions:\n<Subject-1> carries a courier bag.\n\ndetailed description:\n"
        "[Shot 1] <Subject-1> runs in. [Shot 2] <Subject 1> stops.\n\nOverall Soundscape:\n"
        "Fast footsteps on concrete.",
        # Only subjects 2 and 3, in any spelling; the text's own music is kept.
        "summary:\n[reference generation] <subject 2> confronts <Subject_3>.\n\n"
        "detailed_description:\n[Shot 1] <Subject-2> points at <Subject-3>. [Shot 2] "
        "<Subject-3> raises his hands.\n\noverall-soundscape: Alarms blare.\n\n"
        "NON_DIEGETIC_MUSIC:\nA low synth drone.",
        # No subject: the scene alone.
        "The camera pans slowly across the empty laboratory.",
        "<Subject-4> waits.\n\nOVERALL_SOUNDSCAPE:\nRain on the skylight.",
    ]
    wire = {
        "references": LAB,
        "overall_soundscape": HUM,
        "segments": [{"prompt": text, "duration_s": 6} for text in segments],
    }
    def lab(n: int) -> str:
        return f"<Background> is the background shown in <Picture {n}>. {LAB[4]['description']}"

    expected = [
        (
            f"subject_definitions:\n<Subject-4> is the character shown in <Picture 1>. "
            f"{LAB[3]['description']}\n{lab(2)}\n\n"
            "summary:\n[reference generation] <Subject-4> explores.\n\n"
            f"retention_analysis:\n<Subject-4> (appears in [Shot 1]): fully_preserved - {PERSON}.\n"
            f"<Background> (appears in [Shot 1]): fully_preserved - {SCENE}.\n\n"
            "detailed_description:\n[Shot 1] <Subject-4> is walking through the labratory "
            "<Background>, exploring and looking around.\n\n"
            f"overall_soundscape:\n{HUM}\n\nnon_diegetic_music:\nN/A",
            ["Subject-4", "Background"],
        ),
        (
            f"Subject Definitions:\n<Subject-1> carries a courier bag.\n"
            f"<Subject-1> is the character shown in <Picture 1>. {LAB[0]['description']}\n{lab(2)}\n\n"
            f"retention_analysis:\n<Subject-1> (appears in [Shot 1], [Shot 2]): fully_preserved - {PERSON}.\n"
            f"<Background>: fully_preserved - {SCENE}.\n\n"
            "detailed description:\n[Shot 1] <Subject-1> runs in. [Shot 2] <Subject 1> stops.\n\n"
            f"Overall Soundscape:\nFast footsteps on concrete.\n{HUM}\n\n"
            "non_diegetic_music:\nN/A",
            ["Subject-1", "Background"],
        ),
        (
            f"subject_definitions:\n<Subject-2> is the character shown in <Picture 1>. "
            f"{LAB[1]['description']}\n<Subject-3> is the character shown in <Picture 2>. "
            f"{LAB[2]['description']}\n{lab(3)}\n\n"
            "summary:\n[reference generation] <subject 2> confronts <Subject_3>.\n\n"
            f"retention_analysis:\n<Subject-2> (appears in [Shot 1]): fully_preserved - {PERSON}.\n"
            f"<Subject-3> (appears in [Shot 1], [Shot 2]): fully_preserved - {PERSON}.\n"
            f"<Background>: fully_preserved - {SCENE}.\n\n"
            "detailed_description:\n[Shot 1] <Subject-2> points at <Subject-3>. [Shot 2] "
            "<Subject-3> raises his hands.\n\n"
            f"overall-soundscape: Alarms blare.\n{HUM}\n\n"
            "NON_DIEGETIC_MUSIC:\nA low synth drone.",
            ["Subject-2", "Subject-3", "Background"],
        ),
        (
            f"subject_definitions:\n{lab(1)}\n\n"
            f"retention_analysis:\n<Background>: fully_preserved - {SCENE}.\n\n"
            "detailed_description:\nThe camera pans slowly across the empty laboratory.\n\n"
            f"overall_soundscape:\n{HUM}\n\nnon_diegetic_music:\nN/A",
            ["Background"],
        ),
        (
            f"subject_definitions:\n<Subject-4> is the character shown in <Picture 1>. "
            f"{LAB[3]['description']}\n{lab(2)}\n\n"
            f"retention_analysis:\n<Subject-4>: fully_preserved - {PERSON}.\n"
            f"<Background>: fully_preserved - {SCENE}.\n\n"
            f"detailed_description:\n<Subject-4> waits.\n\nOVERALL_SOUNDSCAPE:\nRain on the skylight.\n{HUM}\n\n"
            "non_diegetic_music:\nN/A",
            ["Subject-4", "Background"],
        ),
    ]
    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        got = calls(job, wire)
        for index, (text, attached) in enumerate(expected):
            check(f"{job.__name__} segment {index + 1} text", got[index][0], text)
            check(f"{job.__name__} segment {index + 1} references", got[index][1], attached)
            check(f"{job.__name__} segment {index + 1} warnings", got[index][2], [])
    coffee(wire)


def coffee(lab: dict[str, Any]) -> None:
    """The coffee-shop prompt as a long_form segment, then a one-line prompt."""
    references = [
        {"name": "Subject 1", "kind": "background", "description": "A coffee shop with a brick wall."},
        {"name": "Subject 2", "kind": "character", "description": "A fluffy white Samoyed."},
        {"name": "Subject 3", "kind": "character", "description": "A young blonde woman.",
         "voice": "sha256:" + "c" * 64},
        {"name": "Subject 4", "kind": "character", "description": "A young man in a hoodie."},
    ]
    got = calls(
        h3.LongFormInput,
        {
            "references": references,
            "segments": [
                {"prompt": COFFEE, "duration_s": 15},
                {"prompt": "<Subject 3> waves goodbye to <Subject 2>.", "duration_s": 5},
            ],
        },
    )
    follows = ("<Audio 1>: reference - <Subject 3>'s voice follows <Audio 1>'s timbre and delivery without "
               "copying the original signal.\n")
    # The text defines <Subject 1>, <Subject 2> (each citing its picture) and the voice <Audio 1>, and
    # writes a retention line for each subject: the template adds only <Subject 3> and <Subject 4>'s
    # pictures, and joins each reference's description to the text's own definition.
    first = (
        COFFEE.replace("wooden coffee table.\n<Subject 2>", "wooden coffee table. A coffee shop with a brick wall.\n<Subject 2>", 1)
        .replace("a curved tail.\n<Subject 3>", "a curved tail. A fluffy white Samoyed.\n<Subject 3>", 1)
        .replace("vocal layer.\n", "vocal layer.\n<Subject 3> is the character shown in <Picture 3>. A young blonde woman.\n"
                 "<Subject 4> is the character shown in <Picture 4>. A young man in a hoodie.\n", 1)
    )
    check("coffee-shop segment: the text is kept and its sections gain the references", got[0][0], first)
    check(
        "coffee-shop segment: each named reference, the scene first, the voice after its character",
        got[0][1:],
        (
            ["Subject 1", "Subject 2", "Subject 3", "Subject 3's voice", "Subject 4"],
            ["Segment 1: <Video 1>, <Video 2> match no reference; left as written."],
        ),
    )
    check(
        "one-line segment: the scene, its two characters and the voice, in request order",
        got[1],
        (
            "subject_definitions:\n"
            "<Subject 1> is the background shown in <Picture 1>. A coffee shop with a brick wall.\n"
            "<Subject 2> is the character shown in <Picture 2>. A fluffy white Samoyed.\n"
            "<Subject 3> is the character shown in <Picture 3>. A young blonde woman.\n"
            "<Audio 1> is the voice-timbre reference for <Subject 3>.\n\n"
            f"retention_analysis:\n<Subject 1>: fully_preserved - {SCENE}.\n"
            f"<Subject 2>: fully_preserved - {PERSON}.\n"
            f"<Subject 3>: fully_preserved - {PERSON}.\n"
            f"{follows}\n"
            "detailed_description:\n<Subject 3> waves goodbye to <Subject 2>.\n\nnon_diegetic_music:\nN/A",
            ["Subject 1", "Subject 2", "Subject 3", "Subject 3's voice"],
            [],
        ),
    )
    # Global <Picture N> tokens name references too, renumbered for the child's own set.
    (renumbered,) = calls(
        h3.LongFormInput,
        {**lab, "segments": [{"prompt": "[Shot 1] <Picture 3> nods. <Picture 9> waves.", "duration_s": 5}]},
    )
    check(
        "global <Picture 3> is Subject-3, the child's <Picture 1>; an unknown token stays",
        (renumbered[0].split("\n\n")[2], renumbered[1], renumbered[2]),
        (
            "detailed_description:\n[Shot 1] <Picture 1> nods. <Picture 9> waves.",
            ["Subject-3", "Background"],
            ["Segment 1: <Picture 9> match no reference; left as written."],
        ),
    )
    (orphan,) = calls(
        h3.LongFormInput,
        {"references": LAB[:2], "segments": [{"prompt": "Dust drifts.", "duration_s": 5}]},
    )
    check(
        "no scene and no character named: H3 needs an image, so every reference is sent",
        orphan[1:],
        (
            ["Subject-1", "Subject-2"],
            ["Segment 1 names no character and there is no background, so it receives all 2 of its image and audio references."],
        ),
    )


def retention_defaults() -> None:
    references = [
        {"name": "Hero", "kind": "character", "description": "A courier in a teal coat."},
        {"name": "Depot", "kind": "background", "description": "A wet train platform."},
        {"name": "Voice", "kind": "audio", "audio": "sha256:" + "c" * 64},
    ]
    wire: dict[str, Any] = {
        "references": references,
        "segments": [{
            "prompt": "detailed_description:\n[Shot 1] <Hero> speaks with <Voice> at <Depot>.",
            "duration_s": 5,
        }],
    }
    labels = ["<Hero> (appears in [Shot 1])", "<Depot> (appears in [Shot 1])", "<Audio 1>"]
    defaults = [
        "fully_preserved - keep the face, hair, and outfit of the referenced character. "
        "Ignore the white background.",
        f"fully_preserved - {SCENE}.",
        "partially_copy - <Audio 1> is copied into the target audio, with other sounds around it.",
    ]
    authored = "reference - retain only the authored details, including the blue backdrop."
    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        name = job.__name__
        for label, payload, expected in (
            ("character default; background and audio unchanged", wire, defaults),
            (
                "authored retention overrides remain verbatim",
                {**wire, "references": [
                    {**ref, "retention_analysis": authored} for ref in references
                ]},
                [authored] * 3,
            ),
        ):
            prompt = calls(job, payload)[0][0]
            retention = prompt.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0]
            check(
                f"{name}: {label}", retention,
                "\n".join(f"{prefix}: {text}" for prefix, text in zip(labels, expected, strict=True)),
            )


def style() -> None:
    look = "Soft watercolor animation with muted pastel colors."
    kite = [
        {"name": "Kite", "kind": "character", "description": "A red paper kite."},
        {"name": "Beach", "kind": "background", "description": "An empty sandy beach."},
    ]
    texts = [
        "summary:\n<Kite> climbs.\n\ndetailed_description:\n[Shot 1] <Kite> climbs.",
        "<Kite> dips toward the sand.",
    ]
    wire: dict[str, Any] = {
        "references": kite,
        "style": look,
        "segments": [{"prompt": text, "duration_s": 5} for text in texts],
    }
    definitions = (
        "subject_definitions:\n<Kite> is the character shown in <Picture 1>. A red paper kite.\n"
        "<Beach> is the background shown in <Picture 2>. An empty sandy beach.\n\n"
    )
    retention = f"<Kite>{{}}: fully_preserved - {PERSON}.\n<Beach>: fully_preserved - {SCENE}.\n\n"
    styled = [
        f"{definitions}summary:\n[reference generation] <Kite> climbs.\n\nretention_analysis:\n"
        + retention.format(" (appears in [Shot 1])")
        + f"detailed_description:\n{look}\n[Shot 1] <Kite> climbs.\n\nnon_diegetic_music:\nN/A",
        f"{definitions}retention_analysis:\n{retention.format('')}"
        f"detailed_description:\n{look}\n<Kite> dips toward the sand.\n\nnon_diegetic_music:\nN/A",
    ]
    own = "<Kite> spins.\n\nStyle: Grainy black-and-white film."

    def prompts(job: Any, payload: dict[str, Any]) -> list[str]:
        return [call[0] for call in calls(job, payload)]

    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        name = job.__name__
        check(f"{name}: style opens the description in both segments", prompts(job, wire), styled)
        check(
            f"{name}: no style, no opening line",
            prompts(job, {key: value for key, value in wire.items() if key != "style"}),
            [text.replace(f"{look}\n", "") for text in styled],
        )
        check(
            f"{name}: a segment's own style section opens its description instead",
            prompts(job, {**wire, "segments": [{"prompt": own, "duration_s": 5}]}),
            [f"{definitions}retention_analysis:\n{retention.format('')}detailed_description:\n"
             "Grainy black-and-white film.\n<Kite> spins.\n\nnon_diegetic_music:\nN/A"],
        )
        extra = {
            **wire,
            "mood": "calm",
            "segments": [{**wire["segments"][0], "camera": "static"}, wire["segments"][1]],
            "references": [{**kite[0], "pose": "standing"}, kite[1]],
        }
        check(f"{name}: unknown fields are ignored, not refused", prompts(job, extra), styled)
        try:
            calls(job, {**wire, "segments": [{"prompt": "x", "duraton_s": 5}]})
        except msgspec.ValidationError as exc:
            check(f"{name}: a misspelled duration_s still refuses", "duration_s" in str(exc), True)
        else:
            raise AssertionError("a segment without duration_s was accepted")


async def references() -> None:
    supplied = ImageAsset("sha256:" + "a" * 64)
    generated = ImageAsset("sha256:" + "b" * 64)
    hero = StoryReference("Hero", "character", "An adult woman in a teal coat.", image=supplied)
    depot = StoryReference("Depot", "background", "A wet train platform.")
    made: list[str] = []

    async def generate(reference: StoryReference) -> ImageAsset:
        made.append(reference.name)
        return generated

    check("a supplied image is reused", await resolve_reference_images([hero], generate), {"Hero": supplied})
    scene_prompt = image_prompt(StoryReference("Depot", "background", "A wet train platform."))
    for forbidden in ("no people", "human figures", "silhouettes", "human-like subjects"):
        check(f"scene generation explicitly excludes {forbidden}", forbidden in scene_prompt.lower(), True)
    mixed = await resolve_reference_images([hero, depot], generate)
    check("only the missing image is generated", (mixed["Depot"] is generated, made), (True, ["Depot"]))
    prepared = ImageAsset("sha256:" + "c" * 64)
    removed: list[str] = []

    async def prepare_character(reference: StoryReference) -> ImageAsset:
        removed.append(reference.name)
        return prepared

    scene = StoryReference("Scene", "background", image=supplied)
    voice = AudioAsset("sha256:" + "d" * 64)
    audio = StoryReference("Voice", "audio", audio=voice)
    missing_character = StoryReference("Courier", "character", "An adult courier in red.")
    made.clear()
    cleaned = await resolve_reference_images(
        [hero, scene, depot, missing_character, audio], generate, prepare_character,
    )
    check("only supplied characters lose their background",
          (cleaned["Hero"] is prepared, cleaned["Scene"] is supplied,
           cleaned["Depot"] is generated, cleaned["Courier"] is generated,
           cleaned["Voice"] is voice, removed, made),
          (True, True, True, True, True, ["Hero"], ["Depot", "Courier"]))
    # Run 1559: names as people type them normalise; only an invisible or duplicate name refuses.
    typed = request(
        list[StoryReference],
        [
            {"name": " subject  1 ", "kind": "character", "description": "A tall courier."},
            {"name": "<Dr. Zoë O'Neil>", "kind": "character", "description": "A surgeon."},
        ],
    )
    check("names normalise", [ref.name for ref in typed], ["subject 1", "Dr. Zoë O'Neil"])
    for names in (["<>"], [" "], ["Subject 1", "SUBJECT-1"]):
        try:
            validate_references(
                request(list[StoryReference], [{"name": n, "kind": "background", "description": "A place."} for n in names])
            )
        except (msgspec.ValidationError, InvalidRequest):
            print(f"  ok   {names} refuses")
        else:
            raise AssertionError(f"{names} accepted")
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def failing(reference: StoryReference) -> ImageAsset:
        if reference.name == "Depot":
            await entered.wait()
            raise ValueError("generation failed")
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    try:
        await resolve_reference_images([depot, StoryReference("Other", "background", "Garden")], failing)
    except ValueError:
        check("a failed generation cancels its siblings", cancelled.is_set(), True)
    else:
        raise AssertionError("generation failure swallowed")


async def reference_order() -> None:
    """se-066: `references` come back in request order, whichever is made first."""
    supplied = {name: ImageAsset("sha256:" + digit * 64) for name, digit in (("Ann", "1"), ("Cal", "3"))}
    refs = [
        StoryReference("Ann", "character", image=supplied["Ann"]),
        StoryReference("Ben", "character", "An adult man in a red coat."),
        StoryReference("Cal", "character", image=supplied["Cal"]),
    ]
    made_images = {name: ImageAsset("sha256:" + digit * 64) for name, digit in (("Ann", "a"), ("Ben", "b"), ("Cal", "c"))}
    gates = {name: asyncio.Event() for name in made_images}
    made: list[str] = []

    async def finish(name: str) -> SimpleNamespace:
        await gates[name].wait()
        made.append(name)
        return SimpleNamespace(image=made_images[name])

    async def remove_background(*, image: ImageAsset) -> SimpleNamespace:
        return await finish(next(name for name, asset in supplied.items() if asset is image))

    async def generate_reference(**_: Any) -> SimpleNamespace:
        return await finish("Ben")

    # The managed callees are module names `_create_references` reads per call.
    fakes: dict[str, Callable[..., Awaitable[SimpleNamespace]]] = {
        "remove_background": remove_background, "generate_reference": generate_reference,
    }
    real = {name: getattr(h3, name) for name in fakes}
    for name, fake in fakes.items():
        setattr(h3, name, fake)
    try:
        attempt = fake_attempt("h3-reference-order")
        shown: list[ImageAsset] = []
        running = asyncio.create_task(h3._create_references(
            fake_context(), refs, fake_telemetry(attempt), fake_outputs(attempt), shown,
            style="", describe_characters=False,
        ))
        for name in ("Cal", "Ben"):
            gates[name].set()
            while name not in made:
                await asyncio.sleep(0)
        for _ in range(5):
            await asyncio.sleep(0)
        check("a reference made early waits for every earlier one", [p.label for p in attempt.published], [])
        gates["Ann"].set()
        images, _, _, _ = await running
    finally:
        for name, callee in real.items():
            setattr(h3, name, callee)
    check("made last to first", made, ["Cal", "Ben", "Ann"])
    check("references are published in request order",
          [p.label for p in attempt.published], ["Reference: Ann", "Reference: Ben", "Reference: Cal"])
    check("the returned references keep request order", shown, [made_images[n] for n in ("Ann", "Ben", "Cal")])
    check("each name holds its own image", images, made_images)


def opening_cuts() -> None:
    refs = [StoryReference("Ann", "character", "A woman."), StoryReference("Ben", "character", "A man."),
            StoryReference("Cal", "character", "A boy."), StoryReference("Hall", "background", "A hall.")]
    ending = "detailed_description:\n[Shot 1] <Cal> waits.\n[Shot 2] <Ann> and <Ben> talk in <Hall>."
    cut = "detailed_description:\n[Shot 1] Hard cut to <Cal> alone in <Hall>.\n[Shot 2] <Ann> enters."
    walk = "detailed_description:\n[Shot 1] <Cal> walks up to <Ann> and <Ben>."
    same = "Cut to a close-up of <Ann>."
    got = opening_cut_warnings([ending, cut, ending, walk, ending, same], refs, [False, True] * 3)
    check("an opening cut to an absent character warns, once", len(got), 1)
    check("the warning names the cut-to character and the remedy",
          ("<Cal>" in got[0], "context_frames: false" in got[0], got[0].startswith("Segment 2")),
          (True, True, True))
    check("no warning without a carried picture",
          opening_cut_warnings([ending, cut], refs, [False, False]), [])
    check("<Picture N> tokens name characters too",
          len(opening_cut_warnings([ending, "Cut to <Picture 3>."], refs, [False, True])), 1)


def retention_lines(text: str) -> list[str]:
    return text.split("retention_analysis:\n", 1)[1].split("\n\n", 1)[0].splitlines()


def definition_lines(text: str) -> list[str]:
    return text.split("subject_definitions:\n", 1)[1].split("\n\n", 1)[0].splitlines()


async def videos() -> None:
    """Videos are named references: attached only where the text names them, by name first,
    else as the global `<Video N>`, and renumbered per call in request order."""
    clip = lambda digit: "sha256:" + digit * 64  # noqa: E731
    refs = [{"name": "Ann", "kind": "character", "description": "A woman in red."},
            {"name": "Ben", "kind": "character", "description": "A man in blue."},
            {"name": "Video 1", "kind": "video", "video": clip("1")},
            {"name": "Video 2", "kind": "video", "video": clip("2")},
            {"name": "Spin Kick", "kind": "video", "video": clip("3")}]
    motion = ("<Video 1> (motion, choreography and camera movement): attribute_transfer - its "
              "movements, timing and camera work are performed by <Ann> and <Ben>.")
    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        name = job.__name__
        got = calls(job, {"references": refs, "segments": [
            {"prompt": "<Ann> spars with <Ben> as in <Video 2>.", "duration_s": 5},
            {"prompt": "<Ben> copies <video_1>, then <Video 2>.", "duration_s": 5},
            {"prompt": "<Ann> lands <Spin Kick>; <Ben> repeats <Video 2>.", "duration_s": 5},
            {"prompt": "retention_analysis:\n<Video 2> (camera): weak_reference - the dolly only.\n\n"
                       "<Ann> walks as in <Video 2>.", "duration_s": 5},
            {"prompt": "<Ann> waves.", "duration_s": 5},
        ]})
        check(f"{name}: a segment naming only <Video 2> sends it as its <Video 1>",
              (got[0][1], "as in <Video 1>." in got[0][0], retention_lines(got[0][0])[-1],
               definition_lines(got[0][0])),
              (["Ann", "Ben", "Video 2"], True, motion,
               ["<Ann> is the character shown in <Picture 1>, whose movements come from <Video 1>. A woman in red.",
                "<Ben> is the character shown in <Picture 2>, whose movements come from <Video 1>. A man in blue.",
                "<Video 1> is the motion and camera reference for <Ann> and <Ben>."]))
        check(f"{name}: two videos, numbered in the order the text names them",
              (got[1][1], "copies <Video 1>, then <Video 2>." in got[1][0]),
              (["Ben", "Video 1", "Video 2"], True))
        check(f"{name}: one clip reused in two segments; a named clip keeps its name",
              (got[2][1], "<Ann> lands <Spin Kick>; <Ben> repeats <Video 1>." in got[2][0],
               "<Spin Kick> is the motion and camera reference in <Video 2> for <Ann> and <Ben>."
               in definition_lines(got[2][0])),
              (["Ann", "Ben", "Video 2", "Spin Kick"], True, True))
        check(f"{name}: an authored clip retention line replaces the default",
              [line for line in retention_lines(got[3][0]) if "Video" in line],
              ["<Video 1> (camera): weak_reference - the dolly only."])
        check(f"{name}: a segment naming no video sends none", got[4][1], ["Ann"])
    fallback = calls(h3.LongFormInput, {"references": [refs[3], {"name": "Hum", "kind": "audio",
                                                                 "audio": clip("4")},
                                                       refs[0]],
                                        "segments": [{"prompt": "<Hum> plays.", "duration_s": 5}]})
    check("the no-picture fallback never attaches a video", fallback[0][1], ["Hum", "Ann"])
    try:
        calls(h3.LongFormInput, {"references": [*refs, {"name": "Flip", "kind": "video", "video": clip("5")}],
                                 "segments": [{"prompt": "<Ann>: <Video 1> <Video 2> <Spin Kick> <Flip>.",
                                               "duration_s": 5}]})
    except InvalidRequest:
        print("  ok   four videos in one call refuse")
    else:
        raise AssertionError("four videos in one call accepted")
    # Direct ref2va: the same references, cleaning and describing off.
    attempt = fake_attempt("h3-videos")
    named = [{**ref, "image": "sha256:" + digit * 64} for ref, digit in zip(refs[:2], "ab", strict=True)]
    payload = request(h3.ReferenceRenderInput, {
        "prompt": "<Ann> spars with <Ben> as in <Spin Kick>.", "duration_s": 5,
        "references": [*named, refs[4]], "clean_references": False, "describe_references": False})
    sent, assets, sound, _, warnings = await h3._direct_reference_assets(
        fake_context(), payload, Assets(), fake_outputs(attempt), fake_telemetry(attempt))
    check("ref2va: pictures, then the clip, labeled per kind",
          [(assets.info(i).kind, assets.info(i).label) for i in range(len(assets))],
          [("image", "Picture 1"), ("image", "Picture 2"), ("video", "Video 1")])
    check("ref2va: the clip's motion line, and a short description warns once",
          (retention_lines(sent.prompt)[-1], warnings),
          (motion, ["Segment 1: <Spin Kick> guides a description of 8 words. H3 copies a clip's own "
                    "people and setting unless the description establishes each character, the "
                    "background and the beats in order; MiniMax's guide asks 350-500 words."]))
    guide = calls(h3.LongFormInput, {"references": [
        refs[0], {**refs[3], "use_audio": True}, {"name": "Line", "kind": "audio", "audio": clip("4")},
        refs[1], {"name": "Dojo", "kind": "background", "description": "A dojo."}], "segments": [
        {"prompt": "summary:\n<Ann> fights <Ben>.\n\ndetailed_description:\n[Shot 1] In <Dojo>, <Ann> "
                   "spars with <Ben> as in <Video 2>; its sound is <Audio 1>, then <Line> plays.",
         "duration_s": 5},
        {"prompt": "summary:\n[video editing] <Ann> replaces the woman in <Video 2>.\n\n"
                   "detailed_description:\n[Shot 1] <Ann> stands in <Video 2>'s field.", "duration_s": 5},
        {"prompt": "summary:\n\ndetailed_description:\n[Shot 1] <Ann> waves.", "duration_s": 5},
    ]})
    check("request order, a use_audio clip's sound labeled just before it, motion bound per character",
          (guide[0][1], definition_lines(guide[0][0]), retention_lines(guide[0][0])[1:3],
           "[reference generation + audio reuse] <Ann> fights <Ben>." in guide[0][0],
           "its sound is <Audio 1>, then <Line> plays." in guide[0][0]),
          (["Ann", "Video 2", "Line", "Ben", "Dojo"],
           ["<Ann> is the character shown in <Picture 1>, whose movements come from <Video 1>. A woman in red.",
            "<Video 1> is the motion and camera reference for <Ann> and <Ben>.",
            "<Audio 1> is the synchronized audio track of <Video 1>.",
            "<Audio 2> is the supplied audio reference for <Line>.",
            "<Ben> is the character shown in <Picture 2>, whose movements come from <Video 1>. A man in blue.",
            "<Dojo> is the background shown in <Picture 3>. A dojo."],
           ["<Audio 1>: partially_copy - the synchronized audio track of <Video 1> is copied into the "
            "target audio, with other sounds around it.",
            "<Video 1> (motion, choreography and camera movement): attribute_transfer - its movements, "
            "timing and camera work are performed by <Ann> and <Ben> in <Dojo>."],
           True, True))
    check("an edit source is the source video, kept, with no motion binding",
          ([line for line in definition_lines(guide[1][0]) if "Video" in line or "<Ann>" in line],
           [line for line in retention_lines(guide[1][0]) if line.startswith("<Video")],
           "[video editing + audio reuse] <Ann> replaces" in guide[1][0]),
          (["<Ann> is the character shown in <Picture 1>. A woman in red.",
            "<Video 1> is the source video.", "<Audio 1> is the synchronized audio track of <Video 1>."],
           ["<Video 1> (source video editing): fully_preserved - its framing, setting and subjects are "
            "kept except where the description changes them."],
           True))
    check("an empty summary gains the task type", "summary:\n[reference generation]\n\n" in guide[2][0], True)
    check("every declared clip is named, so none warns",
          [warning for row in guide for warning in row[2] if "never named" in warning], [])
    unnamed = calls(h3.LongFormInput, {"references": refs, "segments": [
        {"prompt": "<Ann> waves.", "duration_s": 5}, {"prompt": "<Ben> waves.", "duration_s": 5}]})
    check("an unnamed clip warns once each", [row[2] for row in unnamed],
          [[f"<{name}> is never named in a prompt, so H3 never sees it; name the clip where its motion "
            "applies." for name in ("Video 1", "Video 2", "Spin Kick")], []])
    check("a clip keeps its sound only with use_audio", (sound, h3.video_audio(
        request(list[StoryReference], [{**refs[4], "use_audio": True}, refs[3]]))), ((False,), (True, False)))


def voices() -> None:
    """Audio follows MiniMax's guide (§2.4, §4.2, §5.4): a character's `voice` is a timbre
    `reference` reusing the text's `(Sx)`; standalone audio is `partially_copy` unless told."""
    refs = [{"name": "Ann", "kind": "character", "description": "A woman in red.",
             "voice": "sha256:" + "1" * 64},
            {"name": "Line", "kind": "audio", "audio": "sha256:" + "2" * 64},
            {"name": "Theme", "kind": "audio", "audio": "sha256:" + "3" * 64,
             "retention_analysis": "reference - only the tempo of <Theme> guides the score."},
            {"name": "Beach", "kind": "background", "description": "An empty beach."}]
    got = calls(h3.LongFormInput, {"references": refs, "segments": [
        {"prompt": "summary:\n[reference generation] <Ann> speaks.\n\ndetailed_description:\n"
                   "[Shot 1] <Ann> (S1) says <d>[English] Hello there.</d> Then <Line> plays.",
         "duration_s": 5},
        {"prompt": "summary:\n[reference generation] <Ann> hums.\n\n<Ann> hums over <Theme>.",
         "duration_s": 5},
        {"prompt": "A gull calls over <Line>.", "duration_s": 5},
    ]})
    follows = ("<Audio 1>: reference - <Ann>'s voice follows <Audio 1>'s timbre and delivery "
               "without copying the original signal.")
    check("a voice follows its character, reusing the text's speaker ID",
          (got[0][1], "<Audio 1> is the voice-timbre reference for <Ann> (S1)." in definition_lines(got[0][0]),
           follows in retention_lines(got[0][0])),
          (["Ann", "Ann's voice", "Line", "Beach"], True, True))
    check("standalone audio is copied as is by default",
          "<Audio 2>: partially_copy - <Audio 2> is copied into the target audio, with other sounds around it."
          in retention_lines(got[0][0]), True)
    check("the summary gains both audio task types",
          "[reference generation + audio reference + audio reuse]" in got[0][0], True)
    check("without a speaker ID in the text, none is invented",
          "<Audio 1> is the voice-timbre reference for <Ann>." in definition_lines(got[1][0]), True)
    check("an authored retention line is kept",
          ("<Audio 2>: reference - only the tempo of <Theme> guides the score." in retention_lines(got[1][0]),
           "[reference generation + audio reference]" in got[1][0]), (True, True))
    check("a voice never attaches without its character", "Ann's voice" in got[2][1], False)
    try:
        validate_references(request(list[StoryReference], [
            {"name": "Rain", "kind": "audio", "audio": "sha256:" + "4" * 64, "voice": "sha256:" + "5" * 64}]))
    except InvalidRequest:
        print("  ok   only a character takes a voice")
    else:
        raise AssertionError("a voice on audio accepted")


def writer() -> None:
    """describe_references writes what a call with a described clip lacks of the guide format,
    and nothing else."""
    refs = request(list[StoryReference], [
        {"name": "Ann", "kind": "character", "description": "A woman in a red coat."},
        {"name": "Ben", "kind": "character", "description": "A man in a blue suit."},
        {"name": "Hall", "kind": "background", "description": "A marble hall."},
        {"name": "Kick", "kind": "video", "video": "sha256:" + "5" * 64},
    ])
    clips = {"Kick": "PEOPLE:\nP1: left - tall\nP2: right - short\nBEATS:\n0-2 s: P1 kicks P2; P2 falls back."}
    short = "summary:\n<Ann> kicks <Ben>.\n\ndetailed_description:\n[Shot 1] In <Hall>, <Ann> kicks <Ben> as in <Kick>."
    (call,) = compile_segments([short], refs)
    asked = writer_request(short, call.references, clips, "Film noir.", "")
    check("a short call with a described clip asks the writer, with the looks, the clip and the request",
          (asked is not None and "<Ann>: a character. Look: A woman in a red coat." in asked
           and "P1 kicks P2" in asked and short in asked and "Style: Film noir." in asked), True)
    check("no clip description, no writing", writer_request(short, call.references, {}, "", ""), None)
    long = " ".join(["word"] * SHORT_DESCRIPTION)
    full = ("subject_definitions:\n<Ann>'s movements come from the kicker in <Kick>.\n<Ben>'s movements come "
            f"from the one kicked in <Kick>.\n\ndetailed_description:\n[Shot 1] <Ann> <Ben> <Hall> {long}\n\n"
            "overall_soundscape:\nRain.")
    check("a call already in full is left alone", writer_request(full, call.references, clips, "", ""), None)
    description = "[Shot 1] In <Hall>, <Ann> faces <Ben> and kicks him; <Ben> falls back. " + long
    reply = (f"ROLES:\n<Ann>: P1, the kicker\n<Ben>: P2, the one kicked\n\nDESCRIPTION:\n{description}\n\n"
             "SOUNDSCAPE:\nShoes squeak on marble.")
    text, warnings = apply_written(short, reply, refs, call.references, clips, "")
    check("the reply fills the roles, the description and the soundscape",
          (text, warnings),
          ("subject_definitions:\n<Ann> is the character whose appearance comes from <Picture 1> and whose "
           "movements come from the kicker in <Kick>.\n<Ben> is the character whose appearance comes from "
           f"<Picture 2> and whose movements come from the one kicked in <Kick>.\n\nsummary:\n<Ann> kicks <Ben>.\n\ndetailed_description:\n{description}"
           "\n\noverall_soundscape:\nShoes squeak on marble.\n", []))
    (bound,) = compile_segments([text], refs)
    check("a character the text defines gets no template line; its description joins the text's",
          definition_lines(bound.prompt),
          ["<Ann> is the character whose appearance comes from <Picture 1> and whose movements come from "
           "the kicker in <Kick>. A woman in a red coat.",
           "<Ben> is the character whose appearance comes from <Picture 2> and whose movements come from "
           "the one kicked in <Kick>. A man in a blue suit.",
           "<Hall> is the background shown in <Picture 3>. A marble hall.",
           "<Kick> is the motion and camera reference in <Video 1> for <Ann> and <Ben>."])
    kept, warned = apply_written(short, reply.replace("<Ben>", "him"), refs, call.references, clips, "")
    check("a description that drops a named character is not used",
          "[Shot 1] In <Hall>, <Ann> kicks <Ben> as in <Kick>." in kept, True)
    unplaced, _ = apply_written(short, reply.replace("In <Hall>, ", ""), refs, call.references, clips, "")
    check("a place told only in words is still named: the description opens in it",
          "detailed_description:\nThe video takes place in <Hall>.\n[Shot 1] <Ann> faces" in unplaced, True)
    check("the warning names the dropped character",
          warned, ["The written description was not used: it never names <Ben>; the text's own is sent."])
    own = short + "\n\noverall_soundscape:\nWind."
    check("the text's own soundscape stays",
          apply_written(own, reply, refs, call.references, clips, "")[0].endswith("overall_soundscape:\nWind."), True)


if __name__ == "__main__":
    writer()
    asyncio.run(videos())
    voices()
    opening_cuts()
    one_offs()
    long_form()
    retention_defaults()
    style()
    music()
    asyncio.run(references())
    asyncio.run(reference_order())
    print("H3 prompts: PASS")
