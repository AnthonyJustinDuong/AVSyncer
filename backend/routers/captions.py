"""Caption workflow: upload one video, transcribe, edit, and burn captions."""
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from models.schemas import (
    CaptionCue,
    CaptionExportRequest,
    CaptionProject,
    CaptionSaveRequest,
    CaptionStyle,
    CaptionWord,
)
from routers.sync import _write_session_json, sessions
from services.caption_renderer import render_caption_overlay_video
from services.caption_timing import word_highlight_intervals
from services.ffmpeg_runner import (
    burn_overlay_with_progress,
    burn_subtitles_with_progress,
    ffmpeg_filter_available,
    get_duration,
    get_video_metadata,
)
from services.take_detector import transcribe

router = APIRouter()

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
MAX_CUE_WORDS = 6
MAX_CUE_DURATION = 4.0
MAX_WORD_GAP = 0.9
MIN_CAPTION_MAX_WIDTH = 0.12
MAX_CAPTION_MAX_WIDTH = 0.96
PREVIEW_EDGE_INSET = 0.02
CAPTION_SPLIT_LLM_MODEL = os.environ.get("CAPTION_SPLIT_LLM_MODEL", "gpt-5.4")
CAPTION_SPLIT_LLM_REASONING_EFFORT = os.environ.get("CAPTION_SPLIT_LLM_REASONING_EFFORT", "medium")


def _save_upload(upload: UploadFile, dest: str) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)


def _file_url(session_id: str, filename: str) -> str:
    return f"/api/files/{session_id}/{filename}"


def _caption_project(session_id: str, data: dict) -> CaptionProject:
    project = data.get("caption_project")
    if not project:
        raise HTTPException(status_code=404, detail="Caption project not found")
    return CaptionProject(**project)


def _clean_words(raw_words: list[dict]) -> list[CaptionWord]:
    words: list[CaptionWord] = []
    for raw in raw_words:
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        words.append(CaptionWord(
            id=str(uuid.uuid4()),
            start=float(raw.get("start", 0.0)),
            end=float(raw.get("end", 0.0)),
            text=text,
        ))
    words.sort(key=lambda w: (w.start, w.end))
    return words


def _build_cues(words: list[CaptionWord]) -> list[CaptionCue]:
    mode = (os.environ.get("CAPTION_SPLIT_MODE") or "").strip().lower()
    if not mode:
        mode = "llm" if os.environ.get("OPENAI_API_KEY") else "deterministic"
    if mode == "llm":
        try:
            cues = _build_cues_llm(words)
            if cues:
                return cues
        except Exception as e:
            print(f"[captions] LLM cue split failed, using deterministic fallback: {e}", flush=True)
    elif mode not in {"deterministic", "heuristic"}:
        print(f"[captions] unknown CAPTION_SPLIT_MODE={mode}; using deterministic fallback", flush=True)
    return _build_cues_deterministic(words)


def _build_cues_deterministic(words: list[CaptionWord]) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    current: list[CaptionWord] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        cues.append(CaptionCue(
            id=str(uuid.uuid4()),
            start=current[0].start,
            end=current[-1].end,
            words=current,
        ))
        current = []

    for word in words:
        if current:
            gap = word.start - current[-1].end
            duration = word.end - current[0].start
            ended_sentence = _ends_sentence(current[-1].text)
            if gap > MAX_WORD_GAP or ended_sentence or duration > MAX_CUE_DURATION or len(current) >= MAX_CUE_WORDS:
                flush()
        current.append(word)
    flush()
    return cues


def _build_cues_from_spans(words: list[CaptionWord], spans: list[tuple[int, int]]) -> list[CaptionCue]:
    cues: list[CaptionCue] = []
    for start, end in spans:
        cue_words = words[start:end]
        if not cue_words:
            continue
        cues.append(CaptionCue(
            id=str(uuid.uuid4()),
            start=cue_words[0].start,
            end=cue_words[-1].end,
            words=cue_words,
        ))
    return cues


def _build_cues_llm(words: list[CaptionWord]) -> list[CaptionCue]:
    if not words:
        return []
    if not os.environ.get("OPENAI_API_KEY"):
        return []
    raw = _call_caption_split_llm(words)
    spans = _parse_caption_spans(raw, len(words))
    return _build_cues_from_spans(words, spans)


def _caption_split_json_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cue_spans": {
                "type": "array",
                "description": "Caption cues in chronological order, preferably 5-6 words each, covering every word exactly once.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_word": {
                            "type": "integer",
                            "description": "Inclusive index of the first word in this cue.",
                        },
                        "end_word": {
                            "type": "integer",
                            "description": "Exclusive index immediately after the final word in this cue.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for this sentence or phrase boundary.",
                        },
                    },
                    "required": ["start_word", "end_word", "reason"],
                },
            },
        },
        "required": ["cue_spans"],
    }


def _call_caption_split_llm(words: list[CaptionWord]) -> str:
    from openai import OpenAI

    client = OpenAI()
    transcript = "\n".join(
        f"{i}: {w.start:.3f}-{w.end:.3f} {w.text}"
        for i, w in enumerate(words)
    )
    schema = _caption_split_json_schema()
    system_prompt = (
        "You split word-level video transcripts into readable caption cues. "
        "Target 5-6 words per cue. Four or seven words are acceptable when they make "
        "the sentence read naturally, but avoid cues longer than seven words unless "
        "there is no clean phrase boundary. Each distinct sentence must be its own cue. "
        "Do not merge different sentences into one cue. Split long sentences into "
        "multiple 5-6 word cues at natural clause or phrase boundaries. Keep short "
        "sentences as short cues rather than merging them with another sentence. "
        "Keep filler words with the sentence they belong to. "
        "Return cue spans that cover every word exactly once, in chronological order. "
        "Use [start_word, end_word) indices exactly as provided."
    )
    user_prompt = (
        "Create logical caption cue boundaries for this timestamped word transcript. "
        "Prefer 5-6 words per cue. Different sentences must be different cues:\n\n"
        f"{transcript}"
    )
    inputs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text_format = {
        "format": {
            "type": "json_schema",
            "name": "caption_cue_split",
            "strict": True,
            "schema": schema,
        },
    }

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=CAPTION_SPLIT_LLM_MODEL,
            input=inputs,
            reasoning={"effort": CAPTION_SPLIT_LLM_REASONING_EFFORT},
            text=text_format,
        )
        return _extract_response_text(response)

    try:
        response = client.chat.completions.create(
            model=CAPTION_SPLIT_LLM_MODEL,
            messages=inputs,
            reasoning_effort=CAPTION_SPLIT_LLM_REASONING_EFFORT,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "caption_cue_split",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
    except TypeError:
        response = client.chat.completions.create(
            model=CAPTION_SPLIT_LLM_MODEL,
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


def _parse_caption_spans(raw: str, word_count: int) -> list[tuple[int, int]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"caption split returned invalid JSON: {e}") from e

    spans: list[tuple[int, int]] = []
    for item in data.get("cue_spans") or []:
        try:
            start = int(item["start_word"])
            end = int(item["end_word"])
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("caption split returned a span with invalid indices")
        spans.append((start, end))

    if not spans:
        raise RuntimeError("caption split returned no cue spans")
    cursor = 0
    for start, end in spans:
        if start != cursor or end <= start or end > word_count:
            raise RuntimeError("caption split spans must cover every word exactly once")
        cursor = end
    if cursor != word_count:
        raise RuntimeError("caption split did not cover every word")
    return spans


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!"))


def _normalize_style(style: CaptionStyle) -> CaptionStyle:
    style.max_width = max(MIN_CAPTION_MAX_WIDTH, min(MAX_CAPTION_MAX_WIDTH, style.max_width))
    half_width = style.max_width / 2
    style.x = max(
        PREVIEW_EDGE_INSET + half_width,
        min(1 - PREVIEW_EDGE_INSET - half_width, style.x),
    )
    style.y = max(PREVIEW_EDGE_INSET, min(1 - PREVIEW_EDGE_INSET, style.y))
    style.font_size = max(8, min(180, style.font_size))
    style.outline_width = max(0.0, min(12.0, style.outline_width))
    style.shadow_opacity = max(0.0, min(1.0, style.shadow_opacity))
    style.shadow_blur = max(0.0, min(24.0, style.shadow_blur))
    style.shadow_offset = max(0.0, min(24.0, style.shadow_offset))
    if style.align not in {"left", "center", "right"}:
        style.align = "center"
    if style.highlight_mode not in {"progressive", "active_word", "pop_word"}:
        style.highlight_mode = "progressive"
    return style


def _validate_cues(cues: list[CaptionCue]) -> None:
    if not cues:
        raise HTTPException(status_code=400, detail="No captions to export")
    visible = False
    for cue in cues:
        if cue.end <= cue.start:
            raise HTTPException(status_code=400, detail="Caption cue has invalid timing")
        if any(w.text.strip() for w in cue.words):
            visible = True
    if not visible:
        raise HTTPException(status_code=400, detail="No caption text to export")


@router.post("/captions/upload", response_model=CaptionProject)
async def upload_caption_video(video: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOADS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_ext = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    video_filename = f"caption_input{video_ext}"
    video_path = os.path.join(session_dir, video_filename)
    _save_upload(video, video_path)

    try:
        duration = get_duration(video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    created_at = datetime.now(timezone.utc).isoformat()
    project = CaptionProject(
        session_id=session_id,
        created_at=created_at,
        video_url=_file_url(session_id, video_filename),
        duration=duration,
        cues=[],
        style=CaptionStyle(),
    )
    sessions[session_id] = {
        "kind": "caption",
        "caption_path": video_path,
        "caption_video_filename": video_filename,
        "duration": duration,
        "created_at": created_at,
        "caption_project": project.model_dump(),
        "caption_words": None,
    }
    _write_session_json(session_id)
    return project


@router.post("/captions/transcribe", response_model=CaptionProject)
async def transcribe_caption_video(
    session_id: str = Form(...),
    force_retranscribe: bool = Form(False),
):
    session = sessions.get(session_id)
    if not session or session.get("kind") != "caption":
        raise HTTPException(status_code=404, detail="Caption session not found")

    video_path = session["caption_path"]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Caption video not found")

    try:
        project = _caption_project(session_id, session)
        raw_words = session.get("caption_words")
        if force_retranscribe or not raw_words:
            raw_words = transcribe(video_path)
            session["caption_words"] = raw_words
            words = _clean_words(raw_words)
            project.cues = _build_cues(words)
        elif not project.cues:
            words = _clean_words(raw_words)
            project.cues = _build_cues(words)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    session["caption_project"] = project.model_dump()
    _write_session_json(session_id)
    return project


@router.post("/captions/save", response_model=CaptionProject)
async def save_caption_project(req: CaptionSaveRequest):
    session = sessions.get(req.session_id)
    if not session or session.get("kind") != "caption":
        raise HTTPException(status_code=404, detail="Caption session not found")

    project = _caption_project(req.session_id, session)
    _validate_cues(req.cues)
    project.cues = req.cues
    project.style = _normalize_style(req.style)
    session["caption_project"] = project.model_dump()
    _write_session_json(req.session_id)
    return project


@router.get("/captions/sessions", response_model=list[CaptionProject])
async def list_caption_sessions():
    items: list[CaptionProject] = []
    for sid, data in sessions.items():
        if data.get("kind") != "caption" or not data.get("caption_project"):
            continue
        items.append(CaptionProject(**data["caption_project"]))
    items.sort(key=lambda p: p.created_at, reverse=True)
    return items


@router.post("/captions/export")
async def export_caption_video(req: CaptionExportRequest):
    session = sessions.get(req.session_id)
    if not session or session.get("kind") != "caption":
        raise HTTPException(status_code=404, detail="Caption session not found")

    video_path = session["caption_path"]
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Caption video not found")

    _validate_cues(req.cues)
    style = _normalize_style(req.style)
    session_dir = os.path.dirname(video_path)
    ass_path = os.path.join(session_dir, "captions.ass")
    overlay_path = os.path.join(session_dir, "caption_overlay.mov")
    out_path = os.path.join(session_dir, "caption_export.mp4")
    download_url = _file_url(req.session_id, "caption_export.mp4")
    duration = float(session.get("duration") or get_duration(video_path))

    def generate():
        try:
            project = _caption_project(req.session_id, session)
            project.cues = req.cues
            project.style = style
            session["caption_project"] = project.model_dump()
            _write_session_json(req.session_id)
            if ffmpeg_filter_available("subtitles"):
                metadata = get_video_metadata(video_path)
                _write_ass_file(ass_path, req.cues, style, int(metadata["width"]), int(metadata["height"]))
                for pct in burn_subtitles_with_progress(video_path, ass_path, out_path, duration):
                    yield f"data: {json.dumps({'progress': pct})}\n\n"
            else:
                for pct in render_caption_overlay_video(video_path, overlay_path, req.cues, style, duration):
                    yield f"data: {json.dumps({'progress': pct})}\n\n"
                for pct in burn_overlay_with_progress(video_path, overlay_path, out_path, duration):
                    mapped_pct = min(100, 20 + round(pct * 0.8))
                    yield f"data: {json.dumps({'progress': mapped_pct})}\n\n"
            yield f"data: {json.dumps({'done': True, 'download_url': download_url})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _write_ass_file(path: str, cues: list[CaptionCue], style: CaptionStyle, video_width: int, video_height: int) -> None:
    align = {"left": 1, "center": 2, "right": 3}.get(style.align, 2)
    x, margin_l, margin_r = _ass_horizontal_box(style, video_width)
    y = round(style.y * video_height)
    outline = _ass_outline_size(style)
    shadow = _ass_shadow_size(style)
    events: list[str] = []
    for cue in cues:
        if not cue.words or not any(w.text.strip() for w in cue.words):
            continue
        if style.highlight_mode in {"active_word", "pop_word"}:
            events.extend(_ass_active_word_events(
                cue,
                x=x,
                y=y,
                align=align,
                style=style,
                margin_l=margin_l,
                margin_r=margin_r,
                pop=style.highlight_mode == "pop_word",
            ))
        else:
            events.append(_ass_progressive_event(
                cue,
                x=x,
                y=y,
                align=align,
                margin_l=margin_l,
                margin_r=margin_r,
            ))
    content = "\n".join([
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Poppins,{style.font_size},{_ass_color(style.highlight_color)},{_ass_color(style.base_color)},{_ass_color(style.outline_color)},{_ass_color(style.shadow_color, style.shadow_opacity)},-1,0,0,0,100,100,0,0,1,{outline},{shadow},{align},{margin_l},{margin_r},40,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
        "",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _ass_progressive_event(
    cue: CaptionCue,
    x: int,
    y: int,
    align: int,
    margin_l: int,
    margin_r: int,
) -> str:
    parts: list[str] = [f"{{\\an{align}\\pos({x},{y})\\q2}}"]
    for idx, word in enumerate(cue.words):
        duration_cs = max(1, round(max(0.01, word.end - word.start) * 100))
        text = _ass_escape(word.text.strip())
        parts.append(f"{{\\kf{duration_cs}}}{text}")
        if idx < len(cue.words) - 1:
            parts.append(" ")
    return f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},Default,,{margin_l},{margin_r},0,,{''.join(parts)}"


def _ass_active_word_events(
    cue: CaptionCue,
    x: int,
    y: int,
    align: int,
    style: CaptionStyle,
    margin_l: int,
    margin_r: int,
    pop: bool = False,
) -> list[str]:
    base_tags = f"{{\\an{align}\\pos({x},{y})\\q2\\1c{_ass_override_color(style.base_color)}}}"
    base_text = base_tags + _ass_plain_words(cue)
    events = [
        f"Dialogue: 0,{_ass_time(cue.start)},{_ass_time(cue.end)},Default,,{margin_l},{margin_r},0,,{base_text}"
    ]

    intervals = word_highlight_intervals(cue)
    for active in cue.words:
        interval = intervals.get(active.id)
        if not active.text.strip() or not interval or interval.end <= interval.start:
            continue
        parts = [
            f"{{\\an{align}\\pos({x},{y})\\q2\\1c{_ass_override_color(style.highlight_color)}}}"
        ]
        for idx, word in enumerate(cue.words):
            if word.id == active.id:
                word_tags = (
                    r"\alpha&H00&\fscx118\fscy118\t(0,140,\fscx100\fscy100)"
                    if pop else
                    r"\alpha&H00&\fscx100\fscy100"
                )
            else:
                word_tags = r"\alpha&HFF&\fscx100\fscy100"
            parts.append(f"{{{word_tags}}}{_ass_escape(word.text.strip())}")
            if idx < len(cue.words) - 1:
                parts.append(" ")
        events.append(
            f"Dialogue: 1,{_ass_time(interval.start)},{_ass_time(interval.end)},Default,,{margin_l},{margin_r},0,,{''.join(parts)}"
        )
    return events


def _ass_horizontal_box(style: CaptionStyle, video_width: int) -> tuple[int, int, int]:
    max_width = max(MIN_CAPTION_MAX_WIDTH, min(MAX_CAPTION_MAX_WIDTH, style.max_width))
    half_width = max_width / 2
    center = max(
        PREVIEW_EDGE_INSET + half_width,
        min(1 - PREVIEW_EDGE_INSET - half_width, style.x),
    )
    left = max(0.0, min(1.0, center - half_width))
    right = max(left + 0.01, min(1.0, center + half_width))
    if style.align == "left":
        pos_x = left
    elif style.align == "right":
        pos_x = right
    else:
        pos_x = center
    margin_l = round(left * video_width)
    margin_r = round((1 - right) * video_width)
    return round(pos_x * video_width), margin_l, margin_r


def _ass_plain_words(cue: CaptionCue) -> str:
    return " ".join(_ass_escape(word.text.strip()) for word in cue.words)


def _ass_outline_size(style: CaptionStyle) -> str:
    outline = max(0.0, min(12.0, style.outline_width))
    return f"{outline:.1f}"


def _ass_shadow_size(style: CaptionStyle) -> str:
    shadow = max(0.0, min(24.0, style.shadow_offset))
    if style.shadow_opacity <= 0:
        shadow = 0.0
    return f"{shadow:.1f}"


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs == 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ass_color(hex_color: str, opacity: float = 1.0) -> str:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_color.strip())
    value = match.group(1) if match else "ffffff"
    rr = value[0:2]
    gg = value[2:4]
    bb = value[4:6]
    alpha = round((1 - max(0.0, min(1.0, opacity))) * 255)
    return f"&H{alpha:02X}{bb}{gg}{rr}"


def _ass_override_color(hex_color: str) -> str:
    match = re.fullmatch(r"#?([0-9a-fA-F]{6})", hex_color.strip())
    value = match.group(1) if match else "ffffff"
    rr = value[0:2]
    gg = value[2:4]
    bb = value[4:6]
    return f"&H{bb}{gg}{rr}&"


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
