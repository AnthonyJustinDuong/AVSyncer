"""Bad take detection using word-level Whisper transcription + phrase-retake splitting."""
import json
import os
import sys
import tempfile
import time
import uuid

from faster_whisper import WhisperModel

from models.schemas import Partition


def _log(msg: str) -> None:
    print(f"[take_detector] {msg}", flush=True, file=sys.stderr)


# ── Tunables ──────────────────────────────────────────────────────────────
MAX_GAP = 0.3                   # word-to-word silence before a chunk break
MIN_LEN = 0.5                   # drop chunks shorter than this
MAX_CHUNK_DURATION = 30.0       # hard cap on a single chunk's length
LLM_TAKE_DETECTOR_MODEL = "gpt-5.4"
LLM_REASONING_EFFORT = "high"


# Load model once (downloads on first use, ~1.5GB for medium)
_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _log("loading Whisper 'medium' model (first run downloads ~1.5GB from HF Hub)...")
        t0 = time.time()
        # int8 quantization is fast on CPU with minimal accuracy loss
        _model = WhisperModel("medium", device="cpu", compute_type="int8")
        _log(f"model ready in {time.time() - t0:.1f}s")
    return _model


def transcribe(media_path: str) -> list[dict]:
    """
    Run Whisper on the media file and return a flat list of word-level
    timestamps: [{"start": float, "end": float, "text": str}, ...].
    Uses the OpenAI API when OPENAI_API_KEY is set (fast, paid), otherwise
    falls back to local faster-whisper (slow, free). The caller should cache
    this list per session so LLM take detection can re-run without
    re-transcribing.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return _transcribe_openai(media_path)
    return _transcribe_local(media_path)


def _transcribe_local(audio_path: str) -> list[dict]:
    model = _get_model()
    _log(f"transcribing {audio_path} (local faster-whisper, word-level)...")
    t0 = time.time()
    # word_timestamps=True runs forced alignment per segment; ~1.3-1.5x slower
    # than segment-level but gives us the per-word start/end we need.
    segments_gen, info = model.transcribe(audio_path, word_timestamps=True)
    total_duration = float(getattr(info, "duration", 0.0) or 0.0)
    _log(f"audio duration: {total_duration:.1f}s — decoding words...")

    words: list[dict] = []
    last_log = time.time()
    for s in segments_gen:
        for w in (s.words or []):
            words.append({
                "start": float(w.start),
                "end": float(w.end),
                "text": w.word,
            })
        now = time.time()
        if now - last_log >= 2.0:
            pct = (s.end / total_duration * 100.0) if total_duration > 0 else 0.0
            _log(f"  {s.end:.1f}s / {total_duration:.1f}s ({pct:.0f}%) — {len(words)} words")
            last_log = now

    elapsed = time.time() - t0
    _log(f"transcription done: {len(words)} words in {elapsed:.1f}s")
    return words


def _transcribe_openai(media_path: str) -> list[dict]:
    from openai import OpenAI
    from services.ffmpeg_runner import extract_audio_compressed

    _log(f"transcribing {media_path} (OpenAI whisper-1, word-level)...")
    tmp_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(tmp_fd)
    try:
        t0 = time.time()
        extract_audio_compressed(media_path, mp3_path)
        size_mb = os.path.getsize(mp3_path) / 1e6
        _log(f"extracted {size_mb:.1f} MB of audio in {time.time() - t0:.1f}s")
        if size_mb > 24.5:
            raise RuntimeError(f"audio too large for OpenAI API: {size_mb:.1f} MB (25 MB max)")

        t0 = time.time()
        client = OpenAI()
        with open(mp3_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )
        _log(f"OpenAI returned transcription in {time.time() - t0:.1f}s")

        raw_words = getattr(resp, "words", None) or []
        words: list[dict] = []
        for w in raw_words:
            get = w.get if isinstance(w, dict) else lambda k: getattr(w, k)
            words.append({
                "start": float(get("start")),
                "end": float(get("end")),
                "text": get("word"),
            })
        _log(f"transcription done: {len(words)} words")
        return words
    finally:
        if os.path.exists(mp3_path):
            os.unlink(mp3_path)


def decide_partitions_llm(
    words: list[dict],
    total_duration: float,
    model: str = LLM_TAKE_DETECTOR_MODEL,
) -> list[Partition]:
    """
    Ask a reasoning model to identify retake groups from the full timestamped
    word transcript. The model returns word-index spans for repeated/aborted
    takes only; ungrouped transcript regions are treated as one-shot keepers.

    Good take policy is intentionally simple: within every retake group, the
    chronologically last valid take is kept and earlier takes are cut.
    """
    total_duration = max(0.0, float(total_duration))
    if not words:
        if total_duration <= 0:
            return []
        return [_gap_partition(0.0, total_duration)]
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for LLM take detection")

    transcript = _format_words_for_llm(words)
    _log(f"requesting LLM take detection from {model} "
         f"({len(words)} words, reasoning={LLM_REASONING_EFFORT})")
    t0 = time.time()
    raw = _call_take_detector_llm(model, total_duration, transcript)
    _log(f"LLM take detection returned in {time.time() - t0:.1f}s")

    retake_groups = _parse_llm_retake_groups(raw, len(words))
    if retake_groups:
        _log(f"LLM identified {len(retake_groups)} retake groups")
    else:
        _log("LLM identified no retake groups; keeping all one-shot speech")
    return _partitions_from_llm_retake_groups(words, total_duration, retake_groups)


def _format_words_for_llm(words: list[dict]) -> str:
    lines = []
    for i, w in enumerate(words):
        text = str(w.get("text") or "").strip()
        lines.append(f"{i}: {float(w['start']):.3f}-{float(w['end']):.3f} {text}")
    return "\n".join(lines)


def _take_detector_json_schema() -> dict:
    take_span = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start_word": {
                "type": "integer",
                "description": "Inclusive index of the first word in this take.",
            },
            "end_word": {
                "type": "integer",
                "description": "Exclusive index immediately after the final word in this take.",
            },
            "notes": {
                "type": "string",
                "description": "Short reason this span is part of the retake group.",
            },
        },
        "required": ["start_word", "end_word", "notes"],
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "retake_groups": {
                "type": "array",
                "description": (
                    "Only repeated, restarted, or abandoned attempts. Do not include "
                    "normal one-shot transcript sections."
                ),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Brief description of the repeated content.",
                        },
                        "takes": {
                            "type": "array",
                            "description": (
                                "All attempts in chronological order. The backend will keep "
                                "the last valid take and cut the earlier ones."
                            ),
                            "minItems": 2,
                            "items": take_span,
                        },
                    },
                    "required": ["summary", "takes"],
                },
            },
        },
        "required": ["retake_groups"],
    }


def _call_take_detector_llm(model: str, total_duration: float, transcript: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    schema = _take_detector_json_schema()
    system_prompt = (
        "You are an expert video editor finding bad takes in a timestamped transcript. "
        "A bad take is an earlier attempt that is repeated, restarted, abandoned, or "
        "corrected by a later version of substantially the same line. The good take is "
        "always the last valid version in chronological order. Transcript content that "
        "is only said once is a one-shot keeper and must not be included in retake_groups. "
        "Use word indices exactly as provided. Spans are [start_word, end_word), where "
        "end_word is exclusive. Include immediate correction chatter or abandoned words "
        "with the failed take when they should be cut before the successful version. "
        "Prefer under-grouping if repetition is ambiguous or intentional."
    )
    user_prompt = (
        f"Media duration: {total_duration:.3f} seconds\n\n"
        "Return retake groups from this full timestamped word transcript:\n\n"
        f"{transcript}"
    )
    inputs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text_format = {
        "format": {
            "type": "json_schema",
            "name": "take_detection",
            "strict": True,
            "schema": schema,
        },
    }

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            input=inputs,
            reasoning={"effort": LLM_REASONING_EFFORT},
            text=text_format,
        )
        return _extract_response_text(response)

    # Compatibility fallback for older OpenAI SDKs. This keeps the app usable,
    # but upgrading the SDK is preferred because Responses supports reasoning.
    _log("OpenAI SDK has no Responses API; falling back to Chat Completions")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=inputs,
            reasoning_effort=LLM_REASONING_EFFORT,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "take_detection",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
    except TypeError:
        response = client.chat.completions.create(
            model=model,
            messages=[
                inputs[0],
                {
                    "role": "user",
                    "content": (
                        user_prompt
                        + "\n\nReturn only valid JSON matching this schema:\n"
                        + json.dumps(schema)
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )
    return response.choices[0].message.content or "{}"


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts)
    raise RuntimeError("OpenAI response did not include text output")


def _parse_llm_retake_groups(raw: str, word_count: int) -> list[list[tuple[int, int]]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM take detection returned invalid JSON: {e}") from e

    groups: list[list[tuple[int, int]]] = []
    occupied: list[tuple[int, int]] = []
    for group in data.get("retake_groups") or []:
        spans: list[tuple[int, int]] = []
        for take in group.get("takes") or []:
            try:
                start = int(take["start_word"])
                end = int(take["end_word"])
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or end > word_count or start >= end:
                continue
            spans.append((start, end))
        spans = sorted(set(spans))
        if len(spans) < 2:
            continue
        if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
            continue
        if any(_spans_overlap(span, used) for span in spans for used in occupied):
            continue
        groups.append(spans)
        occupied.extend(spans)
    return groups


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _partitions_from_llm_retake_groups(
    words: list[dict],
    total_duration: float,
    retake_groups: list[list[tuple[int, int]]],
) -> list[Partition]:
    chunk_partitions: list[Partition] = []
    covered = [False] * len(words)

    for spans in retake_groups:
        spans.sort(key=lambda s: words[s[0]]["start"])
        group_id = str(uuid.uuid4())
        for take_idx, (start_i, end_i) in enumerate(spans):
            sub = words[start_i:end_i]
            for i in range(start_i, end_i):
                covered[i] = True
            chunk_partitions.append(Partition(
                id=str(uuid.uuid4()),
                start=float(sub[0]["start"]),
                end=float(sub[-1]["end"]),
                text=" ".join(str(w.get("text") or "").strip() for w in sub).strip(),
                group_id=group_id,
                take_index=take_idx,
                keep=(take_idx == len(spans) - 1),
            ))

    cursor = 0
    n = len(words)
    while cursor < n:
        while cursor < n and covered[cursor]:
            cursor += 1
        start = cursor
        while cursor < n and not covered[cursor]:
            cursor += 1
        if start == cursor:
            continue
        for chunk in _merge_words(words[start:cursor], min_len=0.0):
            chunk_partitions.append(Partition(
                id=str(uuid.uuid4()),
                start=float(chunk["start"]),
                end=float(chunk["end"]),
                text=chunk["text"].strip(),
                group_id=str(uuid.uuid4()),
                take_index=0,
                keep=True,
            ))

    chunk_partitions.sort(key=lambda p: p.start)
    return _fill_gaps(chunk_partitions, total_duration)


def _gap_partition(start: float, end: float) -> Partition:
    return Partition(
        id=str(uuid.uuid4()),
        start=float(start),
        end=float(end),
        text="",
        group_id=str(uuid.uuid4()),
        take_index=0,
        keep=True,
    )


def _fill_gaps(chunk_partitions: list[Partition], total_duration: float) -> list[Partition]:
    """
    Insert gap partitions so the returned list fully partitions
    [0, total_duration). Adjacent partitions satisfy end == next.start.
    Minor word-timing overlaps are clamped by snapping each partition's start
    forward to the running cursor.
    """
    eps = 1e-6
    filled: list[Partition] = []
    cursor = 0.0
    for p in chunk_partitions:
        start = max(cursor, p.start)
        end = max(start, p.end)
        if start - cursor > eps:
            filled.append(_gap_partition(cursor, start))
        if end - start > eps:
            filled.append(p.model_copy(update={"start": start, "end": end}))
            cursor = end
        else:
            cursor = max(cursor, end)
    if total_duration - cursor > eps:
        filled.append(_gap_partition(cursor, total_duration))
    elif filled and filled[-1].end < total_duration:
        filled[-1] = filled[-1].model_copy(update={"end": total_duration})
    return filled


def _merge_words(
    words: list[dict],
    max_gap: float = MAX_GAP,
    min_len: float = MIN_LEN,
    max_duration: float = MAX_CHUNK_DURATION,
) -> list[dict]:
    """
    Merge consecutive words into phrase-length chunks. A gap > max_gap ends the
    current chunk; chunks are also capped at max_duration and dropped if they're
    shorter than min_len.

    Each chunk carries its constituent words so the inline-retake splitter
    (which needs per-word timing) can operate without re-aligning.
    """
    if not words:
        return []

    chunks: list[dict] = []
    current = _new_chunk(words[0])

    for w in words[1:]:
        gap = w["start"] - current["end"]
        duration = current["end"] - current["start"]
        if gap <= max_gap and duration < max_duration:
            current["end"] = w["end"]
            current["text"] = (current["text"] + " " + w["text"]).strip()
            current["words"].append(w)
        else:
            if current["end"] - current["start"] >= min_len:
                chunks.append(current)
            current = _new_chunk(w)

    if current["end"] - current["start"] >= min_len:
        chunks.append(current)

    return chunks


def _new_chunk(word: dict) -> dict:
    return {
        "start": word["start"],
        "end": word["end"],
        "text": word["text"].strip(),
        "words": [word],
    }

