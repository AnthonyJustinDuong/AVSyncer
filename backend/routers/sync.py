"""POST /api/sync — upload video + audio, detect offset, produce merged video."""
import json
import os
import uuid
import shutil
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import SyncResponse, SessionInfo, AnalysisResponse
from services.audio_sync import find_sync_offset
from services.ffmpeg_runner import merge_audio_video, get_duration

router = APIRouter()

# In-memory session store: session_id -> {paths + cached info}
sessions: dict[str, dict] = {}

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


def _session_json_path(session_id: str) -> str:
    return os.path.join(UPLOADS_DIR, session_id, "session.json")


def _write_session_json(session_id: str) -> None:
    data = sessions.get(session_id)
    if not data:
        return
    info = {
        "session_id": session_id,
        "kind": data.get("kind", "sync"),
        "created_at": data.get("created_at"),
        "sync": data.get("sync"),
        "analysis": data.get("analysis"),
        "words": data.get("words"),
        "caption_project": data.get("caption_project"),
        "caption_words": data.get("caption_words"),
        "caption_video_filename": data.get("caption_video_filename"),
    }
    with open(_session_json_path(session_id), "w") as f:
        json.dump(info, f)


def load_sessions_from_disk() -> None:
    """Scan uploads/ at startup and rehydrate the in-memory sessions dict."""
    if not os.path.isdir(UPLOADS_DIR):
        return
    for entry in os.listdir(UPLOADS_DIR):
        session_dir = os.path.join(UPLOADS_DIR, entry)
        if not os.path.isdir(session_dir):
            continue
        sj = os.path.join(session_dir, "session.json")
        if not os.path.exists(sj):
            continue
        try:
            with open(sj) as f:
                info = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        sync = info.get("sync") or {}
        analysis = info.get("analysis")
        caption_project = info.get("caption_project")
        caption_video_filename = info.get("caption_video_filename")

        if caption_project and caption_video_filename:
            caption_path = os.path.join(session_dir, caption_video_filename)
            if os.path.exists(caption_path):
                sessions[entry] = {
                    "kind": "caption",
                    "caption_path": caption_path,
                    "caption_video_filename": caption_video_filename,
                    "duration": caption_project.get("duration"),
                    "created_at": info.get("created_at"),
                    "caption_project": caption_project,
                    "caption_words": info.get("caption_words"),
                }
                continue

        synced_path = os.path.join(session_dir, "synced.mp4")
        if not os.path.exists(synced_path):
            continue
        if analysis and not isinstance(analysis.get("partitions"), list):
            # Pre-partition analyses (phrase-level "segments") are incompatible
            # with the new full-partitioning schema; drop them so the user
            # re-runs /analyze. Word-level transcripts below are still reused.
            analysis = None
        sessions[entry] = {
            "kind": "sync",
            "synced_path": synced_path,
            "duration": sync.get("duration"),
            "created_at": info.get("created_at"),
            "sync": sync,
            "analysis": analysis,
            # Pre-migration session.json files stored a phrase-level list under
            # "transcript"; ignore it so the new word-level pipeline re-transcribes
            # on next analyze.
            "words": info.get("words"),
        }


def _save_upload(upload: UploadFile, dest: str) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


@router.post("/sync", response_model=SyncResponse)
async def sync_files(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_ext = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    audio_ext = os.path.splitext(audio.filename or "audio.wav")[1] or ".wav"

    video_path = os.path.join(session_dir, f"input_video{video_ext}")
    audio_path = os.path.join(session_dir, f"input_audio{audio_ext}")
    synced_path = os.path.join(session_dir, "synced.mp4")

    _save_upload(video, video_path)
    _save_upload(audio, audio_path)

    try:
        offset = find_sync_offset(video_path, audio_path)
        merge_audio_video(video_path, audio_path, offset, synced_path)
        duration = get_duration(synced_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    sync_response = SyncResponse(
        session_id=session_id,
        offset_seconds=offset,
        synced_video_url=f"/api/files/{session_id}/synced.mp4",
        duration=duration,
    )

    sessions[session_id] = {
        "kind": "sync",
        "video_path": video_path,
        "audio_path": audio_path,
        "synced_path": synced_path,
        "duration": duration,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sync": sync_response.model_dump(),
        "analysis": None,
        "words": None,
    }
    _write_session_json(session_id)

    return sync_response


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """Return all known sessions with their sync + analysis state, newest first."""
    items: list[SessionInfo] = []
    for sid, data in sessions.items():
        sync = data.get("sync")
        if not sync:
            continue
        analysis_dict = data.get("analysis")
        items.append(SessionInfo(
            session_id=sid,
            created_at=data.get("created_at") or "",
            sync=SyncResponse(**sync),
            analysis=AnalysisResponse(**analysis_dict) if analysis_dict else None,
        ))
    items.sort(key=lambda s: s.created_at, reverse=True)
    return items
