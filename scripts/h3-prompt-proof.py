"""The exact text H3's encoder receives, built by the package's own prompt code; no model.

One-off calls send their prompt verbatim, plus an `N/A` music line when it names none.
long_form segments also gain the global sections, `style` among them, and only the
references they name. Unknown request fields are ignored; a missing required one refuses.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, get_type_hints

import msgspec
from cozy_runtime.author import AudioAsset, ImageAsset, InvalidRequest
from cozy_runtime.models.minimax_h3.official import Task

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "minimax-h3"))
import h3
from story import (
    StoryReference,
    encoder_prompt,
    resolve_reference_images,
    validate_references,
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
    {"name": "Background", "kind": "scene", "description": "An underground laboratory with steel benches."},
]
HUM = "A low hum of laboratory machinery fills the hall."


def request(job: Any, wire: Any) -> Any:
    """Decode the wire as Runtime does; an asset handle stands in for granted bytes."""

    def handle(kind: type, value: Any) -> Any:
        if kind in (AudioAsset, ImageAsset):
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
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

    score = "A gentle string quartet, no vocals."
    scene = {"name": "Beach", "kind": "scene", "description": "An empty beach."}
    prefix = (
        "subject_definitions:\n<Beach> is the scene shown in <Picture 1>. An empty beach.\n\n"
        f"retention_analysis:\n<Beach>: fully_preserved - {SCENE}.\n\n{ONE_LINE}\n\n"
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
    lab = f"<Background> is the scene shown in <Picture 1>. {LAB[4]['description']}"
    expected = [
        (
            f"subject_definitions:\n{lab}\n<Subject-4> is the character shown in <Picture 2>. "
            f"{LAB[3]['description']}\n\n"
            "summary:\n[reference generation] <Subject-4> explores.\n\n"
            f"retention_analysis:\n<Background> (appears in [Shot 1]): fully_preserved - {SCENE}.\n"
            f"<Subject-4> (appears in [Shot 1]): fully_preserved - {PERSON}.\n\n"
            "detailed_description:\n[Shot 1] <Subject-4> is walking through the labratory "
            "<Background>, exploring and looking around.\n\n"
            f"overall_soundscape:\n{HUM}\n\nnon_diegetic_music:\nN/A",
            ["Background", "Subject-4"],
        ),
        (
            f"Subject Definitions:\n<Subject-1> carries a courier bag.\n{lab}\n"
            f"<Subject-1> is the character shown in <Picture 2>. {LAB[0]['description']}\n\n"
            f"retention_analysis:\n<Background>: fully_preserved - {SCENE}.\n"
            f"<Subject-1> (appears in [Shot 1], [Shot 2]): fully_preserved - {PERSON}.\n\n"
            "detailed description:\n[Shot 1] <Subject-1> runs in. [Shot 2] <Subject 1> stops.\n\n"
            f"Overall Soundscape:\nFast footsteps on concrete.\n{HUM}\n\n"
            "non_diegetic_music:\nN/A",
            ["Background", "Subject-1"],
        ),
        (
            f"subject_definitions:\n{lab}\n<Subject-2> is the character shown in <Picture 2>. "
            f"{LAB[1]['description']}\n<Subject-3> is the character shown in <Picture 3>. "
            f"{LAB[2]['description']}\n\n"
            "summary:\n[reference generation] <subject 2> confronts <Subject_3>.\n\n"
            f"retention_analysis:\n<Background>: fully_preserved - {SCENE}.\n"
            f"<Subject-2> (appears in [Shot 1]): fully_preserved - {PERSON}.\n"
            f"<Subject-3> (appears in [Shot 1], [Shot 2]): fully_preserved - {PERSON}.\n\n"
            "detailed_description:\n[Shot 1] <Subject-2> points at <Subject-3>. [Shot 2] "
            "<Subject-3> raises his hands.\n\n"
            f"overall-soundscape: Alarms blare.\n{HUM}\n\n"
            "NON_DIEGETIC_MUSIC:\nA low synth drone.",
            ["Background", "Subject-2", "Subject-3"],
        ),
        (
            f"subject_definitions:\n{lab}\n\n"
            f"retention_analysis:\n<Background>: fully_preserved - {SCENE}.\n\n"
            "The camera pans slowly across the empty laboratory.\n\n"
            f"overall_soundscape:\n{HUM}\n\nnon_diegetic_music:\nN/A",
            ["Background"],
        ),
        (
            f"subject_definitions:\n{lab}\n<Subject-4> is the character shown in <Picture 2>. "
            f"{LAB[3]['description']}\n\n"
            f"retention_analysis:\n<Background>: fully_preserved - {SCENE}.\n"
            f"<Subject-4>: fully_preserved - {PERSON}.\n\n"
            f"<Subject-4> waits.\n\nOVERALL_SOUNDSCAPE:\nRain on the skylight.\n{HUM}\n\n"
            "non_diegetic_music:\nN/A",
            ["Background", "Subject-4"],
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
        {"name": "Subject 1", "kind": "scene", "description": "A coffee shop with a brick wall."},
        {"name": "Subject 2", "kind": "character", "description": "A fluffy white Samoyed."},
        {"name": "Subject 3", "kind": "character", "description": "A young blonde woman."},
        {"name": "Subject 4", "kind": "character", "description": "A young man in a hoodie."},
        {
            "name": "Voice",
            "kind": "audio",
            "description": "The voice timbre of <Subject 3> (S1).",
            "audio": "sha256:" + "c" * 64,
        },
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
    shots = "(appears in [Shot 1], [Shot 2])"
    voice = "<Audio 1> is the supplied audio reference for <Voice>. The voice timbre of <Subject 3> (S1)."
    definitions = (
        "<Subject 1> is the scene shown in <Picture 1>. A coffee shop with a brick wall.\n"
        "<Subject 2> is the character shown in <Picture 2>. A fluffy white Samoyed.\n"
        f"<Subject 3> is the character shown in <Picture 3>. A young blonde woman.\n{voice}\n"
        "<Subject 4> is the character shown in <Picture 4>. A young man in a hoodie.\n"
    )
    retention = (
        f"<Subject 1> (appears in [Shot 1]): fully_preserved - {SCENE}.\n"
        f"<Subject 2> {shots}: fully_preserved - {PERSON}.\n"
        f"<Subject 3> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - {PERSON}.\n"
        "<Audio 1>: fully_preserved - the referenced audio signal and its supplied timing.\n"
        f"<Subject 4> {shots}: fully_preserved - {PERSON}.\n"
    )
    first = COFFEE.replace("vocal layer.\n", "vocal layer.\n" + definitions, 1).replace(
        "original signal.\n", "original signal.\n" + retention, 1
    )
    check("coffee-shop segment: the text is kept and its sections gain the references", got[0][0], first)
    check(
        "coffee-shop segment: each named reference, the scene first, the voice after its character",
        got[0][1:],
        (
            ["Subject 1", "Subject 2", "Subject 3", "Voice", "Subject 4"],
            ["Segment 1: <Video 1>, <Video 2> match no reference; left as written."],
        ),
    )
    check(
        "one-line segment: the scene, its two characters and the voice",
        got[1],
        (
            "subject_definitions:\n"
            "<Subject 1> is the scene shown in <Picture 1>. A coffee shop with a brick wall.\n"
            "<Subject 3> is the character shown in <Picture 2>. A young blonde woman.\n"
            f"{voice}\n"
            "<Subject 2> is the character shown in <Picture 3>. A fluffy white Samoyed.\n\n"
            f"retention_analysis:\n<Subject 1>: fully_preserved - {SCENE}.\n"
            f"<Subject 3>: fully_preserved - {PERSON}.\n"
            "<Audio 1>: fully_preserved - the referenced audio signal and its supplied timing.\n"
            f"<Subject 2>: fully_preserved - {PERSON}.\n\n"
            "<Subject 3> waves goodbye to <Subject 2>.\n\nnon_diegetic_music:\nN/A",
            ["Subject 1", "Subject 3", "Voice", "Subject 2"],
            [],
        ),
    )
    # Global <Picture N> tokens name references too, renumbered for the child's own set.
    (renumbered,) = calls(
        h3.LongFormInput,
        {**lab, "segments": [{"prompt": "[Shot 1] <Picture 3> nods. <Picture 9> waves.", "duration_s": 5}]},
    )
    check(
        "global <Picture 3> is Subject-3, the child's <Picture 2>; an unknown token stays",
        (renumbered[0].split("\n\n")[2], renumbered[1], renumbered[2]),
        (
            "[Shot 1] <Picture 2> nods. <Picture 9> waves.",
            ["Background", "Subject-3"],
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
            ["Segment 1 names no character and there is no scene, so it receives all 2 references."],
        ),
    )


def retention_defaults() -> None:
    references = [
        {"name": "Hero", "kind": "character", "description": "A courier in a teal coat."},
        {"name": "Depot", "kind": "scene", "description": "A wet train platform."},
        {"name": "Voice", "kind": "audio", "audio": "sha256:" + "c" * 64},
    ]
    wire: dict[str, Any] = {
        "references": references,
        "segments": [{
            "prompt": "detailed_description:\n[Shot 1] <Hero> speaks with <Voice> at <Depot>.",
            "duration_s": 5,
        }],
    }
    labels = ["<Depot> (appears in [Shot 1])", "<Hero> (appears in [Shot 1])", "<Audio 1>"]
    defaults = [
        f"fully_preserved - {SCENE}.",
        "fully_preserved - keep the face, hair, and outfit of the referenced character. "
        "Ignore the white background.",
        "fully_preserved - the referenced audio signal and its supplied timing.",
    ]
    authored = "reference - retain only the authored details, including the blue backdrop."
    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        name = job.__name__
        for label, payload, expected in (
            ("character default; scene and audio unchanged", wire, defaults),
            (
                "authored retention overrides remain verbatim",
                {**wire, "references": [
                    {**ref, "retention-analysis": authored} for ref in references
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
        {"name": "Beach", "kind": "scene", "description": "An empty sandy beach."},
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
        "subject_definitions:\n<Beach> is the scene shown in <Picture 1>. An empty sandy beach.\n"
        "<Kite> is the character shown in <Picture 2>. A red paper kite.\n\n"
    )
    retention = f"<Beach>: fully_preserved - {SCENE}.\n<Kite>{{}}: fully_preserved - {PERSON}.\n\n"
    styled = [
        f"{definitions}summary:\n<Kite> climbs.\n\nstyle:\n{look}\n\nretention_analysis:\n"
        + retention.format(" (appears in [Shot 1])")
        + "detailed_description:\n[Shot 1] <Kite> climbs.\n\nnon_diegetic_music:\nN/A",
        f"{definitions}style:\n{look}\n\nretention_analysis:\n{retention.format('')}"
        "<Kite> dips toward the sand.\n\nnon_diegetic_music:\nN/A",
    ]
    own = "<Kite> spins.\n\nStyle: Grainy black-and-white film."

    def prompts(job: Any, payload: dict[str, Any]) -> list[str]:
        return [call[0] for call in calls(job, payload)]

    for job in (h3.LongFormInput, h3.LongFormCutsInput):
        name = job.__name__
        check(f"{name}: style is a section in both segments", prompts(job, wire), styled)
        check(
            f"{name}: no style, no section",
            prompts(job, {key: value for key, value in wire.items() if key != "style"}),
            [text.replace(f"style:\n{look}\n\n", "") for text in styled],
        )
        check(
            f"{name}: a segment's own style is kept",
            prompts(job, {**wire, "segments": [{"prompt": own, "duration_s": 5}]}),
            [f"{definitions}retention_analysis:\n{retention.format('')}{own}\n\n"
             "non_diegetic_music:\nN/A"],
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
    depot = StoryReference("Depot", "scene", "A wet train platform.")
    made: list[str] = []

    async def generate(reference: StoryReference) -> ImageAsset:
        made.append(reference.name)
        return generated

    check("a supplied image is reused", await resolve_reference_images([hero], generate), {"Hero": supplied})
    mixed = await resolve_reference_images([hero, depot], generate)
    check("only the missing image is generated", (mixed["Depot"] is generated, made), (True, ["Depot"]))
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
                request(list[StoryReference], [{"name": n, "kind": "scene", "description": "A place."} for n in names])
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
        await resolve_reference_images([depot, StoryReference("Other", "scene", "Garden")], failing)
    except ValueError:
        check("a failed generation cancels its siblings", cancelled.is_set(), True)
    else:
        raise AssertionError("generation failure swallowed")


if __name__ == "__main__":
    one_offs()
    long_form()
    retention_defaults()
    style()
    music()
    asyncio.run(references())
    print("H3 prompts: PASS")
