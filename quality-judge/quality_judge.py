"""`quality-judge` — the evaluation judge family, as an ordinary cozy-runtime package.

The VLM judge is a PACKAGE, not a job (jobs.md §7, ev-003): request/response scoring on
resident judge models, invoked by eval jobs as ordinary requests. cozy-eval declares five
structural judge protocols (`Judge` / `SoftJudge` / `Transcriber` / `AudioJudge` /
`PreferenceScorer`) and holds all the scoring math; this package holds the MODELS and
nothing else.

Four entrypoints, one per genuinely different read of a resident model:

    judge       HARD. One multi-question prompt per sample, generated as text. The
                caller's tolerant parser turns it into yes/no. Backs `Judge`, and — with
                a different prompt — the fine-detail rubric and the OCR-free VQA lane.
    soft        SOFT. p(yes)/(p(yes)+p(no)) at the FIRST answer position, one prefill per
                question. Backs `SoftJudge`. The yes/no token ids are a fact about the
                TOKENIZER, so they are read here and never on the wire.
    pairwise    BLINDED comparison with the arm-order swap built in: every pair is shown
                (left, right) AND (right, left), and both raw reads come back. Position
                bias is a property of the model, so its control belongs to the model's
                side of the wire.
    transcribe  Speech to text on the resident Whisper. Backs `Transcriber`.

WHAT IS DELIBERATELY NOT HERE — the prompt TEXT. ev-003's checklist says "the package
constructs prompts"; the measurement says otherwise, and the divergence is recorded on the
issue. `build_judge_prompt`, `DETAIL_AXES` and `VQASCORE_TEMPLATE` are versioned WITH
cozy-eval's checklist format, which is locked-core library surface: a second copy here — or
an import that pins a library version into a deployed image — is a second authority whose
drift is invisible in every report either side stamps. What this package DOES construct is
the part that genuinely belongs to the model: the chat template, the image tiling, the
answer-token positions and the decode. Metric text in, raw model reads out.

Requests are BATCH-SHAPED (N calls per request, N typed per-call sub-results) because the
alternative is one attempt per question against a 4.9 GiB resident. A call that fails
carries its reason in `error` and NO answer: the caller lands those items in `unmeasured`,
which is the whole of cozy-eval's honesty contract. A default answer is never invented here.
"""

from __future__ import annotations

import io
import time
from typing import Annotated, Any

import msgspec
from cozy_runtime.author import (
    App,
    AudioAsset,
    Config,
    Context,
    ImageAsset,
    InvalidRequest,
    Loader,
    Model,
    Preflight,
    Telemetry,
    UnsupportedInput,
    uses_components,
)

app = App()

#: The bound on ONE attempt's work. A batch is a convenience for the caller, never a way to
#: buy an unbounded attempt: both the call count and the total decoded media are capped, and
#: the media cap is a cross-call fact no per-field bound can express.
MAX_CALLS = 64
MAX_IMAGES = 256
MAX_AUDIO_SECONDS = 1800.0

#: Whisper's one input rate. Resampling is the package's job — a caller that had to know it
#: would need a resampler in a library whose base install is numpy + msgspec.
ASR_RATE = 16_000


# ------------------------------------------------------------------ wire: the judge lanes


class Frame(msgspec.Struct, forbid_unknown_fields=True):
    """One image in a judged strip.

    A STRUCT around one asset, which is the shape every asset-bearing list in this runtime
    takes (`tiny_h3.ImageReference` is the same thing) and not a decoration: the author
    surface's type walker descends structs and containers of structs, so a bare
    `list[ImageAsset]` field is walked as far as the LIST and its element type is never
    seen — the kernel then decides the schema declares no assets and hydrates nothing.
    Measured, not read: a bare list reached the handler with every asset unhydrated and
    `read_bytes()` refusing. Recorded on ev-003 as a cozy-runtime finding.
    """

    image: ImageAsset


class JudgeCall(msgspec.Struct, forbid_unknown_fields=True):
    """One sample: the images in presentation order, and the prompt to answer about them."""

    id: str
    prompt: str
    images: list[Frame] = []
    """In the order the prompt refers to them. ASSETS, not inline bytes: a request document
    is CONTROL (the supervisor/executor seam caps a frame at 64 KiB and says so), so media
    arrives digest-verified through the delivery grant and is read from the attempt spool.
    An inline-bytes schema would have made this package unservable by construction."""


class JudgeRequest(msgspec.Struct, forbid_unknown_fields=True):
    calls: Annotated[list[JudgeCall], msgspec.Meta(min_length=1, max_length=MAX_CALLS)]
    max_new_tokens: Annotated[int, msgspec.Meta(ge=1, le=1024)] = 256


class JudgeReply(msgspec.Struct):
    id: str
    text: str = ""
    error: str = ""
    """Non-empty means NO answer was produced. The caller records the item unmeasured."""
    seconds: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0


class JudgeResponse(msgspec.Struct):
    replies: list[JudgeReply]
    model_ref: str
    calls: int
    infer_seconds: float


class SoftCall(msgspec.Struct, forbid_unknown_fields=True):
    id: str
    question: str
    images: list[Frame] = []


class SoftRequest(msgspec.Struct, forbid_unknown_fields=True):
    calls: Annotated[list[SoftCall], msgspec.Meta(min_length=1, max_length=MAX_CALLS)]


class SoftReply(msgspec.Struct):
    id: str
    p_yes: float = -1.0
    """In [0, 1]. -1.0 is not a score: it is the sentinel that pairs with a set `error`."""
    error: str = ""
    seconds: float = 0.0


class SoftResponse(msgspec.Struct):
    replies: list[SoftReply]
    model_ref: str
    calls: int
    infer_seconds: float
    yes_tokens: int
    no_tokens: int


class PairCall(msgspec.Struct, forbid_unknown_fields=True):
    """One blinded comparison. The package is never told which side is the candidate."""

    id: str
    prompt: str
    left: list[Frame]
    right: list[Frame]


class PairRequest(msgspec.Struct, forbid_unknown_fields=True):
    calls: Annotated[list[PairCall], msgspec.Meta(min_length=1, max_length=MAX_CALLS)]
    max_new_tokens: Annotated[int, msgspec.Meta(ge=1, le=256)] = 64


class PairReply(msgspec.Struct):
    """BOTH orderings' raw reads. The verdict — and whether the two agree — is the
    caller's, because "A" means a position here and an arm only over there."""

    id: str
    forward: str = ""
    """The read with `left` shown first."""
    swapped: str = ""
    """The read with `right` shown first."""
    error: str = ""
    seconds: float = 0.0


class PairResponse(msgspec.Struct):
    replies: list[PairReply]
    model_ref: str
    calls: int
    infer_seconds: float


# ------------------------------------------------------------------ wire: transcription


class AudioCall(msgspec.Struct, forbid_unknown_fields=True):
    id: str
    samples: AudioAsset
    """Mono float32 little-endian PCM, as an asset. Downmixing is the caller's (it has the
    array); resampling is ours (it needs a resampler the caller's base install does not
    have)."""
    sample_rate: Annotated[int, msgspec.Meta(ge=1000, le=384_000)]
    language: str = ""
    """ISO-639-1 hint. Empty means Whisper detects it, which is the honest default."""


class AudioRequest(msgspec.Struct, forbid_unknown_fields=True):
    calls: Annotated[list[AudioCall], msgspec.Meta(min_length=1, max_length=MAX_CALLS)]


class AudioReply(msgspec.Struct):
    id: str
    text: str = ""
    error: str = ""
    seconds: float = 0.0
    audio_seconds: float = 0.0


class AudioResponse(msgspec.Struct):
    replies: list[AudioReply]
    model_ref: str
    calls: int
    infer_seconds: float


# ------------------------------------------------------------------ preflight


class BatchFacts(msgspec.Struct, frozen=True):
    """The cross-call bound. Metadata only: no bytes are decoded to answer it, so a batch
    that asks for too much work is refused before a single image is parsed."""

    media_count: int


def _preflight_media(count: int, *, field: str) -> BatchFacts:
    if count > MAX_IMAGES:
        raise UnsupportedInput(
            f"{count} images in one request exceeds the {MAX_IMAGES}-image attempt bound: "
            "a batch amortizes a resident model over many samples, it does not buy an "
            "unbounded attempt — split the batch",
            code="capacity",
            fields=[field],
        )
    return BatchFacts(media_count=count)


def preflight_judge(payload: JudgeRequest) -> BatchFacts:
    return _preflight_media(sum(len(c.images) for c in payload.calls), field="calls")


def preflight_soft(payload: SoftRequest) -> BatchFacts:
    return _preflight_media(sum(len(c.images) for c in payload.calls), field="calls")


def preflight_pair(payload: PairRequest) -> BatchFacts:
    return _preflight_media(
        sum(len(c.left) + len(c.right) for c in payload.calls), field="calls"
    )


def preflight_audio(payload: AudioRequest) -> BatchFacts:
    # Preflight reads COUNTS and typed metadata, never bytes (§1.3) — so the duration bound
    # is enforced in the handler, per call, where the samples exist. Saying that out loud
    # is the point: a bound that pretends to run earlier than it can is not a bound.
    return _preflight_media(len(payload.calls), field="calls")


# ------------------------------------------------------------------ plain helpers
# Module code: it touches no component, so it is not Model code (§1.1).


def decode_images(blobs: list[Frame], *, where: str) -> list[Any]:
    """Hydrated image assets -> RGB images. An asset that is not a decodable image is a
    typed REQUEST refusal, not a backend fault: the caller sent it."""
    from PIL import Image, UnidentifiedImageError

    out = []
    for i, blob in enumerate(blobs):
        try:
            image = Image.open(io.BytesIO(blob.image.read_bytes()))
            image.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidRequest(
                f"{where}: image {i} is not decodable ({type(exc).__name__}): "
                f"{str(exc)[:120]}",
                code="unsupported_input",
                fields=[where],
            ) from exc
        out.append(image.convert("RGB"))
    return out


def resample_mono(samples: Any, rate: int, target: int) -> Any:
    """Linear resample of a mono float32 track. Deliberately the cheap one: Whisper's own
    front end is a 16 kHz mel filterbank and the judge lane's inputs are already
    band-limited generator output, so a polyphase kernel would buy accuracy nothing here
    can read — and it would need a dependency this package does not otherwise have."""
    import numpy as np

    if rate == target or samples.size == 0:
        return samples
    n = round(samples.size * target / rate)
    if n <= 0:
        return samples[:0]
    at = np.linspace(0.0, samples.size - 1, n, dtype=np.float64)
    return np.interp(at, np.arange(samples.size, dtype=np.float64), samples).astype(np.float32)


def persist_buffers(module: Any) -> Any:
    """Promote every non-persistent buffer into the module's state_dict.

    NOT a workaround — a construction decision, and the one this runtime forces (§3.2,
    cr-008a): the fill plane moves STORED bytes and refuses a component carrying a buffer
    no checkpoint supplies (`derived_unmaterialized`), because the alternative is serving
    uninitialized memory under a derived name. Re-executing a derived table on the target
    device is cr-008b's staging pass and does not exist yet, so the tables are PRECOMPUTED
    into the artifact instead — the same choice `tiny_h3`'s modulation-table fixture makes. Here
    that is three rope tables (0.5 MiB); the package and the artifact writer call this one
    function, so the topology they agree on cannot drift.
    """
    for sub in module.modules():
        stray = set(getattr(sub, "_non_persistent_buffers_set", ()))
        for name in stray:
            sub._non_persistent_buffers_set.discard(name)
    return module


# ------------------------------------------------------------------ the judge model


class JudgePipeline:
    """The resident VLM plus the processor built from the artifact's own config.

    The processor is NOT weights and has no checkpoint destinations, so it is built from
    the immutable config capability — tokenizer definition, chat template and image
    preprocessing all ride there. cr-008a's artifact carries exactly one non-tensor
    carrier (`Config`), so that is where a checkpoint's non-tensor assets live; if
    TensorFS grows a dedicated one, this moves with no change to the package's shape.
    """

    def __init__(self, config: Config) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForImageTextToText

        mapping = config.mapping()
        model_config = AutoConfig.for_model(**_sub(mapping, "model_config"))
        model = AutoModelForImageTextToText.from_config(model_config, dtype=torch.bfloat16)
        self.components: dict[str, Any] = {"judge": persist_buffers(model.eval())}
        self.processor = _build_vlm_processor(mapping)
        self.answer_tokens = _answer_tokens(self.processor.tokenizer)


class TranscriberPipeline:
    def __init__(self, config: Config) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, GenerationConfig

        mapping = config.mapping()
        model_config = AutoConfig.for_model(**_sub(mapping, "model_config"))
        model = AutoModelForSpeechSeq2Seq.from_config(model_config, dtype=torch.float16)
        model.generation_config = GenerationConfig(**_sub(mapping, "generation_config"))
        self.components: dict[str, Any] = {"asr": persist_buffers(model.eval())}
        self.processor = _build_asr_processor(mapping)


def build_judge(config: Config) -> JudgePipeline:
    return JudgePipeline(config)


def build_transcriber(config: Config) -> TranscriberPipeline:
    return TranscriberPipeline(config)


def _sub(mapping: dict[str, object], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise InvalidRequest(  # a malformed ARTIFACT, surfaced as a construction refusal
            f"artifact config has no {name!r} section", code="artifact_config"
        )
    return dict(value)


def _named(mapping: dict[str, object], key: str) -> Any:
    """The transformers class the ARTIFACT names. Config data selects the class; this
    module hard-codes no model family, exactly as it names no checkpoint."""
    import transformers

    name = str(mapping[key])
    cls = getattr(transformers, name, None)
    if cls is None:
        raise UnsupportedInput(
            f"the artifact names {name!r}, which this release's transformers does not "
            "provide — the image and the artifact disagree about the model family",
            code="artifact_config",
        )
    return cls


def _blob(mapping: dict[str, object], key: str) -> str:
    """A large opaque asset out of the artifact config.

    It is stored zlib-compressed and URL-SAFE base64 encoded, which is not decoration:
    `Config` refuses any string value containing `://` as a source carrier, and a
    tokenizer definition is 6 MB of text that contains `://` as an ordinary BPE merge. The
    heuristic is right about paths and wrong about documents (recorded on ev-003 as a
    cozy-runtime finding); the encoding is what makes an asset an asset rather than a
    string the carrier check has to interpret, and url-safe base64 additionally cannot
    begin with `/`.
    """
    import base64
    import zlib

    return zlib.decompress(base64.urlsafe_b64decode(str(mapping[key]))).decode()


def _tokenizer(mapping: dict[str, object]) -> Any:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    named = str(mapping.get("tokenizer_class") or "")
    cls = _named(mapping, "tokenizer_class") if named else PreTrainedTokenizerFast
    return cls(
        tokenizer_object=Tokenizer.from_str(_blob(mapping, "tokenizer")),
        **_sub(mapping, "tokenizer_config"),
    )


def _build_vlm_processor(mapping: dict[str, object]) -> Any:
    image = _sub(mapping, "image_processor")
    video = _sub(mapping, "video_processor")
    image_cls = _named(image, "image_processor_type")
    video_cls = _named(video, "video_processor_type")
    image.pop("image_processor_type")
    video.pop("video_processor_type")
    return _named(mapping, "processor_class")(
        image_processor=image_cls(**image),
        video_processor=video_cls(**video),
        tokenizer=_tokenizer(mapping),
        chat_template=str(mapping["chat_template"]),
    )


def _build_asr_processor(mapping: dict[str, object]) -> Any:
    features = _sub(mapping, "feature_extractor")
    features_cls = _named(features, "feature_extractor_type")
    features.pop("feature_extractor_type")
    return _named(mapping, "processor_class")(
        feature_extractor=features_cls(**features), tokenizer=_tokenizer(mapping)
    )


#: The spellings a yes/no answer can start with. Summing the variants is what makes p(yes)
#: a read of the ANSWER rather than of one capitalization.
_YES = ("yes", "Yes", " yes", " Yes", "YES")
_NO = ("no", "No", " no", " No", "NO")


def _answer_tokens(tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    def ids(variants: tuple[str, ...]) -> tuple[int, ...]:
        found = [
            encoded[0]
            for v in variants
            if len(encoded := tokenizer.encode(v, add_special_tokens=False)) == 1
        ]
        return tuple(dict.fromkeys(found))

    yes, no = ids(_YES), ids(_NO)
    if not yes or not no:
        raise UnsupportedInput(
            "this tokenizer encodes no single-token yes/no variant, so p(yes) cannot be "
            "read at one position — the soft lane needs a different judge checkpoint",
            code="output_integrity",
        )
    return yes, no


class JudgeModel(Model[JudgePipeline]):
    """The resident vision-language judge. One component, one lease, two reads of it."""

    pipe: JudgePipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(JudgePipeline, factory=build_judge)

    def unload(self, loader: Loader) -> None:
        del self.pipe

    def _inputs(self, images: list[Any], text: str) -> Any:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": text})
        prompt = self.pipe.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False
        )
        kwargs = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        return self.pipe.processor(**kwargs)

    @uses_components("judge")
    def generate(
        self, images: list[Any], text: str, *, max_new_tokens: int
    ) -> tuple[str, int, int]:
        """One greedy generation. Returns (reply, prompt tokens, generated tokens)."""
        import torch

        model = self.pipe.components["judge"]
        inputs = self._inputs(images, text).to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        reply = self.pipe.processor.batch_decode(
            out[:, prompt_tokens:], skip_special_tokens=True
        )[0]
        return str(reply), prompt_tokens, int(out.shape[1] - prompt_tokens)

    @uses_components("judge")
    def p_yes(self, images: list[Any], question: str) -> float:
        """p(yes) / (p(yes) + p(no)) at the FIRST answer position — one prefill, no decode."""
        import torch

        model = self.pipe.components["judge"]
        inputs = self._inputs(images, question).to(model.device)
        with torch.inference_mode():
            # ONE position's logits. The default computes the language head over EVERY
            # position: 1 600 tokens x 2 048 x 151 936 = 497 GFLOP and a 486 MiB tensor,
            # to read one number at the last position. Measured on this card at 14.3 s per
            # soft call before and 1.0 s after, with identical probabilities out.
            logits = model(**inputs, logits_to_keep=1).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=-1)
        yes_ids, no_ids = self.pipe.answer_tokens
        yes = float(probs[list(yes_ids)].sum())
        no = float(probs[list(no_ids)].sum())
        if yes + no <= 0.0:
            # Not a score of zero: the model put no mass on either answer at that
            # position. The caller is told, and records the item unmeasured.
            raise UnsupportedInput(
                "the model put no probability mass on yes or no at the answer position",
                code="output_integrity",
            )
        return yes / (yes + no)


class TranscriberModel(Model[TranscriberPipeline]):
    """The resident speech-to-text model. MIT code and MIT weights (PROVENANCE.md)."""

    pipe: TranscriberPipeline

    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(TranscriberPipeline, factory=build_transcriber)

    def unload(self, loader: Loader) -> None:
        del self.pipe

    @uses_components("asr")
    def transcribe(self, samples: Any, *, language: str) -> str:
        import torch

        model = self.pipe.components["asr"]
        features = self.pipe.processor.feature_extractor(
            samples, sampling_rate=ASR_RATE, return_tensors="pt"
        )
        inputs = features.input_features.to(device=model.device, dtype=model.dtype)
        kwargs: dict[str, Any] = {"task": "transcribe"}
        if language:
            kwargs["language"] = language
        with torch.inference_mode():
            ids = model.generate(inputs, **kwargs)
        text = self.pipe.processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
        return str(text).strip()


# ------------------------------------------------------------------ entrypoints


def _failed(exc: Exception) -> str:
    """The reason a call produced NO answer, in the shape a report's `unmeasured` takes."""
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:240]}"


@app.entrypoint(preflight=preflight_judge)
def judge(
    ctx: Context,
    payload: JudgeRequest,
    facts: Preflight[BatchFacts],
    model: JudgeModel,
    tel: Telemetry,
) -> JudgeResponse:
    del facts  # admission already keyed on it; the handler re-validates nothing
    replies: list[JudgeReply] = []
    total = 0.0
    step = tel.step_callback(len(payload.calls), stage="judge", overall_range=(0.0, 1.0))
    for i, call in enumerate(payload.calls):
        started = time.perf_counter()
        try:
            images = decode_images(call.images, where=f"calls.{i}.images")
            text, prompt_tokens, generated = model.generate(
                images, call.prompt, max_new_tokens=payload.max_new_tokens
            )
        except InvalidRequest:
            raise
        except Exception as exc:
            replies.append(JudgeReply(id=call.id, error=_failed(exc)))
        else:
            replies.append(
                JudgeReply(
                    id=call.id,
                    text=text,
                    seconds=round(time.perf_counter() - started, 4),
                    prompt_tokens=prompt_tokens,
                    generated_tokens=generated,
                )
            )
        total += time.perf_counter() - started
        step(i)
    return JudgeResponse(
        replies=replies,
        model_ref=model.checkpoint_ref,
        calls=len(replies),
        infer_seconds=round(total, 4),
    )


@app.entrypoint(preflight=preflight_soft)
def soft(
    ctx: Context,
    payload: SoftRequest,
    facts: Preflight[BatchFacts],
    model: JudgeModel,
    tel: Telemetry,
) -> SoftResponse:
    del facts
    replies: list[SoftReply] = []
    total = 0.0
    step = tel.step_callback(len(payload.calls), stage="soft", overall_range=(0.0, 1.0))
    for i, call in enumerate(payload.calls):
        started = time.perf_counter()
        try:
            images = decode_images(call.images, where=f"calls.{i}.images")
            p = model.p_yes(images, call.question)
        except InvalidRequest:
            raise
        except Exception as exc:
            replies.append(SoftReply(id=call.id, error=_failed(exc)))
        else:
            replies.append(
                SoftReply(id=call.id, p_yes=p, seconds=round(time.perf_counter() - started, 4))
            )
        total += time.perf_counter() - started
        step(i)
    yes_ids, no_ids = model.pipe.answer_tokens
    return SoftResponse(
        replies=replies,
        model_ref=model.checkpoint_ref,
        calls=len(replies),
        infer_seconds=round(total, 4),
        yes_tokens=len(yes_ids),
        no_tokens=len(no_ids),
    )


@app.entrypoint(preflight=preflight_pair)
def pairwise(
    ctx: Context,
    payload: PairRequest,
    facts: Preflight[BatchFacts],
    model: JudgeModel,
    tel: Telemetry,
) -> PairResponse:
    """Blinded pairwise comparison, run in BOTH orders.

    The swap is not an option the caller may forget: a comparison judge's position bias is
    a property of the model, so the control for it lives on the model's side of the wire
    and every reply carries both reads.
    """
    del facts
    replies: list[PairReply] = []
    total = 0.0
    step = tel.step_callback(len(payload.calls), stage="pairwise", overall_range=(0.0, 1.0))
    for i, call in enumerate(payload.calls):
        started = time.perf_counter()
        try:
            left = decode_images(call.left, where=f"calls.{i}.left")
            right = decode_images(call.right, where=f"calls.{i}.right")
            forward, _, _ = model.generate(
                left + right, call.prompt, max_new_tokens=payload.max_new_tokens
            )
            ctx.raise_if_cancelled()
            swapped, _, _ = model.generate(
                right + left, call.prompt, max_new_tokens=payload.max_new_tokens
            )
        except InvalidRequest:
            raise
        except Exception as exc:
            replies.append(PairReply(id=call.id, error=_failed(exc)))
        else:
            replies.append(
                PairReply(
                    id=call.id,
                    forward=forward,
                    swapped=swapped,
                    seconds=round(time.perf_counter() - started, 4),
                )
            )
        total += time.perf_counter() - started
        step(i)
    return PairResponse(
        replies=replies,
        model_ref=model.checkpoint_ref,
        calls=len(replies),
        infer_seconds=round(total, 4),
    )


@app.entrypoint(preflight=preflight_audio)
def transcribe(
    ctx: Context,
    payload: AudioRequest,
    facts: Preflight[BatchFacts],
    model: TranscriberModel,
    tel: Telemetry,
) -> AudioResponse:
    del facts
    import numpy as np

    replies: list[AudioReply] = []
    total = 0.0
    step = tel.step_callback(len(payload.calls), stage="transcribe", overall_range=(0.0, 1.0))
    for i, call in enumerate(payload.calls):
        started = time.perf_counter()
        seconds = 0.0
        try:
            raw = call.samples.read_bytes()
            seconds = len(raw) / 4.0 / call.sample_rate
            if len(raw) % 4:
                raise InvalidRequest(
                    f"calls.{i}.samples is {len(raw)} B, which is not a whole number of "
                    "float32 samples",
                    code="unsupported_input",
                    fields=[f"calls.{i}.samples"],
                )
            if seconds > MAX_AUDIO_SECONDS:
                raise UnsupportedInput(
                    f"calls.{i}: {seconds:.0f} s exceeds the "
                    f"{MAX_AUDIO_SECONDS:.0f} s per-call bound",
                    code="capacity",
                    fields=[f"calls.{i}.samples"],
                )
            mono = np.frombuffer(raw, dtype="<f4")
            text = model.transcribe(
                resample_mono(mono, call.sample_rate, ASR_RATE), language=call.language
            )
        except InvalidRequest:
            raise
        except Exception as exc:
            replies.append(AudioReply(id=call.id, error=_failed(exc), audio_seconds=seconds))
        else:
            replies.append(
                AudioReply(
                    id=call.id,
                    text=text,
                    seconds=round(time.perf_counter() - started, 4),
                    audio_seconds=round(seconds, 3),
                )
            )
        total += time.perf_counter() - started
        step(i)
    return AudioResponse(
        replies=replies,
        model_ref=model.checkpoint_ref,
        calls=len(replies),
        infer_seconds=round(total, 4),
    )
