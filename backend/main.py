from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import mimetypes
import os
import re

from dotenv import load_dotenv
load_dotenv()

from routers import sync, analysis, export


@asynccontextmanager
async def lifespan(_app: FastAPI):
    sync.load_sessions_from_disk()
    yield


app = FastAPI(title="AV Syncer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploads_dir = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(uploads_dir, exist_ok=True)

# The installed Starlette (0.38) StaticFiles does not honor HTTP Range
# requests, so <video> elements cannot seek inside served files. We serve
# files ourselves with a Range-aware endpoint that replies 206 Partial
# Content when the client asks for a byte range.
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")
_CHUNK_SIZE = 1024 * 1024


@app.get("/api/files/{session_id}/{filename}")
def serve_session_file(session_id: str, filename: str, request: Request):
    # Guard against path traversal: both path components must be plain names.
    if "/" in session_id or "/" in filename or ".." in session_id or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    file_path = os.path.join(uploads_dir, session_id, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    range_header = request.headers.get("range")
    if range_header is None:
        return FileResponse(
            file_path,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"},
        )

    m = _RANGE_RE.fullmatch(range_header.strip())
    if not m:
        raise HTTPException(status_code=416, detail="Invalid Range header")
    start_s, end_s = m.groups()
    if start_s == "" and end_s == "":
        raise HTTPException(status_code=416, detail="Invalid Range header")
    if start_s == "":
        # Suffix range: last N bytes.
        length = int(end_s)
        if length <= 0:
            raise HTTPException(status_code=416, detail="Invalid Range header")
        start = max(0, file_size - length)
        end = file_size - 1
    else:
        start = int(start_s)
        end = int(end_s) if end_s else file_size - 1
    if start > end or start >= file_size:
        raise HTTPException(
            status_code=416,
            detail="Requested range not satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)
    length = end - start + 1

    def iter_file():
        with open(file_path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(
        iter_file(),
        status_code=206,
        media_type=content_type,
        headers=headers,
    )


app.include_router(sync.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(export.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"status": "ok"}
