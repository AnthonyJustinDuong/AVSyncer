"""Helpers for caption word highlight timing."""
from dataclasses import dataclass
from math import isfinite

from models.schemas import CaptionCue

MIN_ACTIVE_WORD_DURATION = 0.18


@dataclass(frozen=True)
class WordHighlightInterval:
    start: float
    end: float


def word_highlight_intervals(
    cue: CaptionCue,
    min_duration: float = MIN_ACTIVE_WORD_DURATION,
) -> dict[str, WordHighlightInterval]:
    """Return visual active-word intervals for pop/whole-word caption styles.

    Transcription word timings can contain zero-duration or very short words. Those
    are easy to miss at preview/export frame rates, so repair only those cues by
    giving each visible word a proportional sequential slot within the cue.
    """
    words = [word for word in cue.words if word.text.strip()]
    if not words:
        return {}

    cue_start = _finite_number(cue.start, words[0].start)
    cue_end = _finite_number(cue.end, words[-1].end)
    if cue_end <= cue_start:
        cue_end = cue_start + max(min_duration, len(words) * min_duration)

    if not _needs_repair(words, min_duration):
        return {
            word.id: WordHighlightInterval(
                _finite_number(word.start, cue_start),
                _finite_number(word.end, cue_end),
            )
            for word in words
        }

    cue_duration = max(0.001, cue_end - cue_start)
    weights = [
        max(min_duration, _finite_number(word.end, cue_start) - _finite_number(word.start, cue_start))
        for word in words
    ]
    total_weight = sum(weights) or len(words)
    intervals: dict[str, WordHighlightInterval] = {}
    cursor = cue_start

    for index, word in enumerate(words):
        if index == len(words) - 1:
            end = cue_end
        else:
            slot = cue_duration * (weights[index] / total_weight)
            end = min(cue_end, cursor + max(0.001, slot))
        intervals[word.id] = WordHighlightInterval(cursor, end)
        cursor = end

    return intervals


def _needs_repair(words, min_duration: float) -> bool:
    previous_start: float | None = None
    for word in words:
        start = _finite_number(word.start, 0.0)
        end = _finite_number(word.end, start)
        if end - start < min_duration:
            return True
        if previous_start is not None and start <= previous_start:
            return True
        previous_start = start
    return False


def _finite_number(value: float, fallback: float) -> float:
    return value if isfinite(value) else fallback
