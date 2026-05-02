from pydantic import BaseModel


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
