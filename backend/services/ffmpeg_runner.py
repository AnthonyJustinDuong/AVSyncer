"""FFmpeg subprocess helpers."""
import json
import os
import subprocess
from functools import lru_cache


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr}")


def extract_audio(video_path: str, out_wav_path: str) -> None:
    """Extract audio track from a video file to a wav."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        out_wav_path,
    ])


def extract_audio_compressed(video_path: str, out_mp3_path: str) -> None:
    """Mono 16 kHz MP3 at 64 kbps — keeps an hour of audio well under the 25 MB API limit."""
    _run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-b:a", "64k",
        "-f", "mp3",
        out_mp3_path,
    ])


def merge_audio_video(video_path: str, audio_path: str, offset_seconds: float, out_path: str) -> None:
    """Replace the video's audio track with external audio, applying time offset."""
    # offset > 0: external audio starts later (delay it)
    # offset < 0: external audio starts earlier (trim leading silence from audio)
    # -movflags +faststart moves the moov atom to the front of the MP4 so the
    # browser can seek over HTTP without downloading the whole file first.
    if offset_seconds >= 0:
        _run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-itsoffset", str(offset_seconds),
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-movflags", "+faststart",
            "-shortest",
            out_path,
        ])
    else:
        # Trim the audio to remove the portion before the video starts
        trim = abs(offset_seconds)
        _run([
            "ffmpeg", "-y",
            "-i", video_path,
            "-ss", str(trim),
            "-i", audio_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-movflags", "+faststart",
            "-shortest",
            out_path,
        ])


def get_duration(file_path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    return float(result.stdout.strip())


def get_video_metadata(file_path: str) -> dict:
    """Return basic video stream metadata."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-of", "json",
            file_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe error: {result.stderr}")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe did not find a video stream")
    stream = streams[0]
    fps = _parse_fps(stream.get("avg_frame_rate")) or _parse_fps(stream.get("r_frame_rate")) or 30.0
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
    }


def _parse_fps(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def _ffmpeg_filters() -> set[str]:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return set()
    filters: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and "->" in parts[2]:
            filters.add(parts[1])
    return filters


def ffmpeg_filter_available(name: str) -> bool:
    return name in _ffmpeg_filters()


def _build_cut_filter(keep_segments: list[tuple[float, float]]) -> str:
    n = len(keep_segments)
    filter_parts = []
    concat_labels = []
    for i, (start, end) in enumerate(keep_segments):
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
        filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
        concat_labels.append(f"[v{i}][a{i}]")
    concat_input = "".join(concat_labels)
    filter_parts.append(f"{concat_input}concat=n={n}:v=1:a=1[outv][outa]")
    return ";".join(filter_parts)


def cut_segments(input_path: str, keep_segments: list[tuple[float, float]], out_path: str) -> None:
    """Cut a video/audio to keep only the specified time ranges."""
    if not keep_segments:
        raise ValueError("No segments to keep")
    _run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", _build_cut_filter(keep_segments),
        "-map", "[outv]",
        "-map", "[outa]",
        "-movflags", "+faststart",
        out_path,
    ])


def cut_segments_with_progress(input_path: str, keep_segments: list[tuple[float, float]], out_path: str):
    """Same as cut_segments but yields integer progress 0–100 as FFmpeg runs."""
    import threading

    if not keep_segments:
        raise ValueError("No segments to keep")

    total_duration = sum(end - start for start, end in keep_segments)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-filter_complex", _build_cut_filter(keep_segments),
        "-map", "[outv]",
        "-map", "[outa]",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    last_pct = 0
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            val = line.split("=", 1)[1]
            if val == "N/A":
                continue
            try:
                elapsed_s = int(val) / 1_000_000
                pct = min(99, int(elapsed_s / total_duration * 100)) if total_duration > 0 else 0
                if pct > last_pct:
                    last_pct = pct
                    yield pct
            except ValueError:
                pass
        elif line == "progress=end":
            yield 100

    proc.wait()
    t.join()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{''.join(stderr_lines)}")


def burn_subtitles_with_progress(input_path: str, ass_path: str, out_path: str, total_duration: float):
    """Burn an ASS subtitle file into a video and yield integer progress 0-100."""
    import threading

    ass_dir = os.path.dirname(ass_path) or None
    ass_filename = os.path.basename(ass_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"subtitles=filename={ass_filename}",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=ass_dir)
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    last_pct = 0
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            val = line.split("=", 1)[1]
            if val == "N/A":
                continue
            try:
                elapsed_s = int(val) / 1_000_000
                pct = min(99, int(elapsed_s / total_duration * 100)) if total_duration > 0 else 0
                if pct > last_pct:
                    last_pct = pct
                    yield pct
            except ValueError:
                pass
        elif line == "progress=end":
            yield 100

    proc.wait()
    t.join()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{''.join(stderr_lines)}")


def encode_png_sequence_to_qtrle(frame_pattern: str, fps: float, out_path: str) -> None:
    """Encode RGBA PNG frames as a QuickTime Animation overlay video."""
    _run([
        "ffmpeg", "-y",
        "-framerate", f"{fps:.3f}",
        "-i", frame_pattern,
        "-c:v", "qtrle",
        "-pix_fmt", "argb",
        out_path,
    ])


def burn_overlay_with_progress(input_path: str, overlay_path: str, out_path: str, total_duration: float):
    """Overlay a transparent video over the input and yield integer progress 0-100."""
    import threading

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-i", overlay_path,
        "-filter_complex", "[0:v][1:v]overlay=0:0:format=auto:alpha=straight:eof_action=pass:repeatlast=0[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-progress", "pipe:1",
        "-nostats",
        out_path,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr_lines: list[str] = []

    def _drain_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    last_pct = 0
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            val = line.split("=", 1)[1]
            if val == "N/A":
                continue
            try:
                elapsed_s = int(val) / 1_000_000
                pct = min(99, int(elapsed_s / total_duration * 100)) if total_duration > 0 else 0
                if pct > last_pct:
                    last_pct = pct
                    yield pct
            except ValueError:
                pass
        elif line == "progress=end":
            yield 100

    proc.wait()
    t.join()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{''.join(stderr_lines)}")
