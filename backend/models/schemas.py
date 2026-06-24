from pydantic import BaseModel, Field


class Partition(BaseModel):
    id: str
    start: float       # seconds — inclusive
    end: float         # seconds — exclusive; partitions are contiguous so end==next.start
    text: str = ""
    group_id: str      # partitions sharing a group_id are retakes of each other
    take_index: int    # 0-based order within group
    keep: bool         # true = this take survives export


class SyncResponse(BaseModel):
    session_id: str
    offset_seconds: float
    synced_video_url: str
    duration: float


class AnalysisResponse(BaseModel):
    session_id: str
    partitions: list[Partition]
    total_duration: float
    audio_url: str


class ExportRequest(BaseModel):
    session_id: str
    partitions: list[Partition]


class ExportResponse(BaseModel):
    download_url: str


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    sync: SyncResponse
    analysis: AnalysisResponse | None = None


class CaptionWord(BaseModel):
    id: str
    start: float
    end: float
    text: str


class CaptionCue(BaseModel):
    id: str
    start: float
    end: float
    words: list[CaptionWord]


class CaptionStyle(BaseModel):
    x: float = 0.5
    y: float = 0.82
    max_width: float = 0.72
    font_size: int = 48
    base_color: str = "#ffffff"
    highlight_color: str = "#ffd34d"
    outline_color: str = "#000000"
    outline_width: float = 4.0
    shadow_color: str = "#000000"
    shadow_opacity: float = 0.82
    shadow_blur: float = 6.0
    shadow_offset: float = 3.0
    align: str = "center"
    highlight_mode: str = "progressive"


class CaptionProject(BaseModel):
    session_id: str
    created_at: str
    video_url: str
    duration: float
    cues: list[CaptionCue]
    style: CaptionStyle = Field(default_factory=CaptionStyle)


class CaptionSaveRequest(BaseModel):
    session_id: str
    cues: list[CaptionCue]
    style: CaptionStyle


class CaptionExportRequest(BaseModel):
    session_id: str
    cues: list[CaptionCue]
    style: CaptionStyle
