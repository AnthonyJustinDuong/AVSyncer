"""POST /api/export — cut video to keep partitions, produce downloadable MP4."""
import os
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.schemas import ExportRequest, ExportResponse
from services.ffmpeg_runner import cut_segments_with_progress
from routers.sync import sessions

router = APIRouter()

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")


@router.post("/export")
async def export_video(req: ExportRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    synced_path = session["synced_path"]
    if not os.path.exists(synced_path):
        raise HTTPException(status_code=404, detail="Synced video not found")

    keep_segments = [(p.start, p.end) for p in req.partitions if p.keep]

    if not keep_segments:
        raise HTTPException(status_code=400, detail="No partitions marked keep")

    keep_segments.sort(key=lambda x: x[0])

    out_path = os.path.join(UPLOADS_DIR, req.session_id, "export.mp4")
    download_url = f"/api/files/{req.session_id}/export.mp4"

    def generate():
        try:
            for pct in cut_segments_with_progress(synced_path, keep_segments, out_path):
                yield f"data: {json.dumps({'progress': pct})}\n\n"
            yield f"data: {json.dumps({'done': True, 'download_url': download_url})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
