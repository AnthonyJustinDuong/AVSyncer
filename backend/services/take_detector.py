"""Bad take detection using word-level Whisper transcription + phrase-retake splitting."""
import os
import re
import sys
import tempfile
import time
import uuid
from difflib import SequenceMatcher

from faster_whisper import WhisperModel

from models.schemas import Partition


def _log(msg: str) -> None:
    print(f"[take_detector] {msg}", flush=True, file=sys.stderr)


# ── Tunables ──────────────────────────────────────────────────────────────
MAX_GAP = 0.3                   # word-to-word silence before a chunk break
MIN_LEN = 0.5                   # drop chunks shorter than this
MAX_CHUNK_DURATION = 30.0       # hard cap on a single chunk's length
INLINE_RETAKE_LOOKAHEAD = 1.5   # max seconds between the bad take and its retake
INLINE_RETAKE_MIN_WORDS = 2     # shortest phrase we'll treat as a retake
INLINE_RETAKE_MAX_WORDS = 6     # we only scan phrases up to this length
INLINE_RETAKE_SIMILARITY = 0.80
RETAKE_WINDOW = 100             # how many subsequent chunks the whole-chunk dedupe checks
DEDUPE_MIN_WORDS = 3            # below this word count, fall back to aborted-prefix only


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


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _is_aborted_prefix(short_text: str, long_text: str, min_words: int = 3, tolerance: float = 0.80) -> bool:
    """
    True if `short_text` looks like an aborted prefix of `long_text`:
    speaker started a sentence, caught a mistake, and restarted from the top.

    Compares `short_text` against the first len(short_text) chars of `long_text`
    so length mismatch (which kills a full-string ratio) doesn't matter.
    """
    if not short_text or not long_text:
        return False
    if len(short_text) >= len(long_text):
        return False
    if len(short_text.split()) < min_words:
        return False
    prefix = long_text[:len(short_text)]
    return SequenceMatcher(None, short_text, prefix).ratio() >= tolerance


def transcribe(media_path: str) -> list[dict]:
    """
    Run Whisper on the media file and return a flat list of word-level
    timestamps: [{"start": float, "end": float, "text": str}, ...].
    Uses the OpenAI API when OPENAI_API_KEY is set (fast, paid), otherwise
    falls back to local faster-whisper (slow, free). The caller should cache
    this list per session so decide_partitions() can re-run cheaply with
    different thresholds.
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


def decide_partitions(
    words: list[dict],
    total_duration: float,
    similarity_threshold: float = 0.75,
) -> list[Partition]:
    """
    Given a flat word-level transcript plus the total media duration, return a
    full partitioning of [0, total_duration) into contiguous Partitions.

    Each Partition is one of:
      - A spoken chunk (merged from consecutive words)
      - An inline bad-take fragment (words the speaker abandoned mid-phrase)
      - A gap (silence / non-speech) between spoken regions

    Chunks that are retakes of each other share a group_id; keep=True only on
    the last take in the group. Gap partitions are singletons with keep=True.
    Pure transformation — fast, deterministic, no model calls.
    """
    total_duration = max(0.0, float(total_duration))
    if not words:
        if total_duration <= 0:
            return []
        return [_gap_partition(0.0, total_duration)]

    # Pass 1: classify the word stream into contiguous kept / inline-dropped
    # spans. Each dropped span carries the word index where its retake starts
    # so we can group the two together later.
    inline_spans = _classify_inline_spans(words)
    dropped_words = sum(s["b"] - s["a"] for s in inline_spans if s["kind"] == "drop")
    if dropped_words:
        kept_spans = sum(1 for s in inline_spans if s["kind"] == "keep")
        _log(f"inline retake detector flagged {dropped_words} bad-take words "
             f"across {kept_spans} kept spans")

    # Pass 2: build chunk dicts. Kept spans are merged via gap-based chunking;
    # dropped spans become a single chunk each (one bad-take fragment).
    chunks: list[dict] = []
    for span in inline_spans:
        sub = words[span["a"]:span["b"]]
        if not sub:
            continue
        if span["kind"] == "keep":
            for m in _merge_words(sub):
                m["kind"] = "keep"
                chunks.append(m)
        else:
            chunks.append({
                "start": sub[0]["start"],
                "end": sub[-1]["end"],
                "text": " ".join(w["text"] for w in sub).strip(),
                "words": sub,
                "kind": "drop",
                "retake_word_idx": span["retake_a"],
            })
    chunks.sort(key=lambda c: c["start"])
    _log(f"produced {len(chunks)} chunks "
         f"({sum(1 for c in chunks if c['kind'] == 'keep')} kept, "
         f"{sum(1 for c in chunks if c['kind'] == 'drop')} inline-dropped)")

    # Pass 3: union-find groups. Two relations are unioned:
    #   - inline drop ↔ its retake chunk (links a bad-take to its replacement)
    #   - whole-chunk retake dedupe on kept chunks (speaker restarted a phrase
    #     after a longer pause)
    n = len(chunks)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    _union_inline_drops(chunks, words, union)
    _union_whole_chunk_retakes(chunks, similarity_threshold, union)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    multi = [g for g in groups.values() if len(g) > 1]
    _log(f"grouped into {len(groups)} partitions worth of takes "
         f"({len(multi)} retake groups with >1 take)")

    chunk_partitions: list[Partition] = []
    for members in groups.values():
        members.sort(key=lambda i: chunks[i]["start"])
        kept_idx = [i for i in members if chunks[i]["kind"] == "keep"]
        keeper = kept_idx[-1] if kept_idx else None
        group_id = str(uuid.uuid4())
        for take_idx, i in enumerate(members):
            c = chunks[i]
            chunk_partitions.append(Partition(
                id=str(uuid.uuid4()),
                start=float(c["start"]),
                end=float(c["end"]),
                text=c["text"].strip(),
                group_id=group_id,
                take_index=take_idx,
                keep=(i == keeper),
            ))
    chunk_partitions.sort(key=lambda p: p.start)

    # Fill the timeline so every second of [0, total_duration) is covered by
    # exactly one partition. Gaps get their own singleton "keep" partitions.
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


def _union_inline_drops(chunks: list[dict], words: list[dict], union) -> None:
    """Union each inline-dropped chunk with its retake chunk (same phrase,
    re-uttered)."""
    for i, c in enumerate(chunks):
        if c["kind"] != "drop":
            continue
        retake_t = words[c["retake_word_idx"]]["start"]
        # The retake is the first kept chunk that starts at (or just after)
        # the retake word. Fall back to the next kept chunk if nothing lines up
        # (e.g. the retake's first word was dropped by _merge_words min_len).
        exact = None
        fallback = None
        for j, c2 in enumerate(chunks):
            if c2["kind"] != "keep":
                continue
            if c2["start"] < c["end"] - 1e-6:
                continue
            if fallback is None:
                fallback = j
            if abs(c2["start"] - retake_t) < 0.05:
                exact = j
                break
        target = exact if exact is not None else fallback
        if target is not None:
            union(i, target)


def _union_whole_chunk_retakes(
    chunks: list[dict],
    threshold: float,
    union,
    window: int = RETAKE_WINDOW,
) -> None:
    """Detect whole-chunk retakes among kept chunks (speaker said the same
    sentence twice with a long pause between) and union their indices."""
    keep_idx = [i for i, c in enumerate(chunks) if c["kind"] == "keep"]
    normalized = {i: _normalize(chunks[i]["text"]) for i in keep_idx}
    word_counts = {i: len(chunks[i].get("words") or chunks[i]["text"].split()) for i in keep_idx}

    for pos, i in enumerate(keep_idx):
        for j in keep_idx[pos + 1:pos + 1 + window]:
            both_long = min(word_counts[i], word_counts[j]) >= DEDUPE_MIN_WORDS
            similar = both_long and _similarity(normalized[i], normalized[j]) >= threshold
            if similar or _is_aborted_prefix(normalized[i], normalized[j]):
                union(i, j)


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


def _classify_inline_spans(
    words: list[dict],
    lookahead_s: float = INLINE_RETAKE_LOOKAHEAD,
    min_phrase_words: int = INLINE_RETAKE_MIN_WORDS,
    max_phrase_words: int = INLINE_RETAKE_MAX_WORDS,
    similarity: float = INLINE_RETAKE_SIMILARITY,
) -> list[dict]:
    """
    Partition the word stream into an ordered list of spans, each tagged as
    "keep" (a surviving run of words) or "drop" (a bad-take fragment the
    speaker abandoned mid-phrase). Dropped spans carry `retake_a`, the word
    index where the corresponding retake begins, so callers can link the two.

    Algorithm: walk the word list. At each position i, try the longest phrase
    first and scan forward within lookahead_s for a similar phrase at j. When
    found, the span [i, j) is an aborted take and the retake begins at j.

    Two retake shapes are detected:
      - Equal-length: [A, B, C] → [A, B, C]  (fuzzy match, threshold=similarity)
      - Incremental: [A, B]    → [A, B, C]  (aborted prefix of a longer restart)
    """
    n = len(words)
    spans: list[dict] = []
    cursor = 0
    i = 0

    while i < n:
        matched = False
        max_len = min(max_phrase_words, n - i)
        for phrase_len in range(max_len, min_phrase_words - 1, -1):
            if i + phrase_len > n:
                continue
            a_words = words[i:i + phrase_len]
            a_text = _normalize(" ".join(w["text"] for w in a_words))
            if not a_text:
                continue
            a_end_t = a_words[-1]["end"]
            for j in range(i + phrase_len, n):
                if words[j]["start"] - a_end_t > lookahead_s:
                    break
                # Equal-length retake
                if j + phrase_len <= n:
                    b_words = words[j:j + phrase_len]
                    b_text = _normalize(" ".join(w["text"] for w in b_words))
                    if b_text and SequenceMatcher(None, a_text, b_text).ratio() >= similarity:
                        if cursor < i:
                            spans.append({"kind": "keep", "a": cursor, "b": i})
                        spans.append({"kind": "drop", "a": i, "b": j, "retake_a": j})
                        cursor = j
                        i = j
                        matched = True
                        break
                # Incremental retake: compare the short phrase at i against
                # progressively longer phrases at j using aborted-prefix logic.
                for b_len in range(phrase_len + 1, max_phrase_words + 1):
                    if j + b_len > n:
                        break
                    b_words = words[j:j + b_len]
                    b_text = _normalize(" ".join(w["text"] for w in b_words))
                    if not b_text:
                        continue
                    if _is_aborted_prefix(a_text, b_text, min_words=min_phrase_words, tolerance=similarity):
                        if cursor < i:
                            spans.append({"kind": "keep", "a": cursor, "b": i})
                        spans.append({"kind": "drop", "a": i, "b": j, "retake_a": j})
                        cursor = j
                        i = j
                        matched = True
                        break
                if matched:
                    break
            if matched:
                break
        if not matched:
            i += 1

    if cursor < n:
        spans.append({"kind": "keep", "a": cursor, "b": n})
    return spans
