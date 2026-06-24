"""POST /api/analyze — detect bad takes in an audio file."""
import os
from fastapi import APIRouter, Form, HTTPException
from models.schemas import AnalysisResponse
from services.take_detector import transcribe, decide_partitions, decide_partitions_llm
from services.ffmpeg_runner import get_duration
from routers.sync import sessions, _write_session_json

router = APIRouter()

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    session_id: str = Form(...),
    similarity_threshold: float = Form(0.75),
    force_retranscribe: bool = Form(False),
    detector: str | None = Form(None),
):
    """
    Analyze the synced audio from an existing session for bad takes.
    Uses the synced video's audio track.

    Transcription is cached per-session; pass force_retranscribe=true to
    rebuild it. Pass detector=llm to use the model-based retake detector, or
    detector=deterministic for the local heuristic. When omitted, the API uses
    TAKE_DETECTOR_MODE, defaulting to llm when OPENAI_API_KEY is configured and
    deterministic otherwise.
    """
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    synced_path = session["synced_path"]
    if not os.path.exists(synced_path):
        raise HTTPException(status_code=404, detail="Synced video not found")

    try:
        # Cache key renamed from "transcript" (phrase-level) to "words" (word-level)
        # so old phrase-level caches are ignored and re-transcribed on next analyze.
        words = session.get("words")
        if force_retranscribe or not words:
            words = transcribe(synced_path)
            session["words"] = words
        duration = session.get("duration") or get_duration(synced_path)
        requested_detector = (detector or os.environ.get("TAKE_DETECTOR_MODE") or "").strip().lower()
        if not requested_detector:
            requested_detector = "llm" if os.environ.get("OPENAI_API_KEY") else "deterministic"

        if requested_detector == "llm":
            partitions = decide_partitions_llm(words, total_duration=duration)
        elif requested_detector in {"deterministic", "heuristic"}:
            partitions = decide_partitions(
                words,
                total_duration=duration,
                similarity_threshold=similarity_threshold,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown detector: {requested_detector}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = AnalysisResponse(
        session_id=session_id,
        partitions=partitions,
        total_duration=duration,
        audio_url=f"/api/files/{session_id}/synced.mp4",
    )

    session["analysis"] = response.model_dump()
    _write_session_json(session_id)

    return response
