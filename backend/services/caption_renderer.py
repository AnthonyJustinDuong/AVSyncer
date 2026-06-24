"""Render caption overlays for FFmpeg builds without text filters."""
import glob
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Iterator

from models.schemas import CaptionCue, CaptionStyle, CaptionWord
from services.ffmpeg_runner import encode_png_sequence_to_qtrle, get_video_metadata
from services.caption_timing import word_highlight_intervals

OVERLAY_FPS = 30.0
PREVIEW_EDGE_INSET = 0.02
MIN_CAPTION_MAX_WIDTH = 0.12
MAX_CAPTION_MAX_WIDTH = 0.96
CAPTION_LINE_HEIGHT = 1.12
CAPTION_POP_DURATION = 0.18
CAPTION_POP_PEAK = 1.18
CAPTION_POP_PEAK_AT = 0.45


@dataclass
class WordLayout:
    word: CaptionWord
    text: str
    width: float
    descent: float
    font: object
    fill_pct: float
    is_active: bool
    active_start: float
    pop_scale: float


@dataclass
class FontLineMetrics:
    ascent: float
    descent: float
    height: float


def render_caption_overlay_video(
    video_path: str,
    overlay_path: str,
    cues: list[CaptionCue],
    style: CaptionStyle,
    total_duration: float,
    progress_start: int = 1,
    progress_end: int = 20,
) -> Iterator[int]:
    """Render transparent caption frames and encode them as a qtrle MOV."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise RuntimeError(
            "Pillow is required for caption export with this FFmpeg build. "
            "Run backend/.venv/bin/pip install -r backend/requirements.txt, then restart dev.sh."
        ) from e

    metadata = get_video_metadata(video_path)
    width = int(metadata["width"])
    height = int(metadata["height"])
    source_fps = float(metadata.get("fps") or OVERLAY_FPS)
    fps = max(12.0, min(OVERLAY_FPS, source_fps))
    frame_count = max(1, math.ceil(total_duration * fps) + 1)
    frame_dir = tempfile.mkdtemp(prefix="caption_frames_")
    font_cache: dict[int, object] = {}
    cue_index = 0
    last_progress = progress_start - 1

    def font_for(size: int):
        size = max(1, int(size))
        if size not in font_cache:
            font_cache[size] = ImageFont.truetype(_caption_font_path(), size=size)
        return font_cache[size]

    try:
        for frame_index in range(frame_count):
            t = frame_index / fps
            while cue_index < len(cues) and t > cues[cue_index].end:
                cue_index += 1
            cue = cues[cue_index] if cue_index < len(cues) and cues[cue_index].start <= t <= cues[cue_index].end else None

            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            if cue:
                _draw_caption(image, cue, t, style, width, height, font_for)

            image.save(os.path.join(frame_dir, f"frame_{frame_index:06d}.png"))
            progress = progress_start + int((frame_index + 1) / frame_count * (progress_end - progress_start))
            if progress > last_progress:
                last_progress = progress
                yield progress

        encode_png_sequence_to_qtrle(os.path.join(frame_dir, "frame_%06d.png"), fps, overlay_path)
        if progress_end > last_progress:
            yield progress_end
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


def _draw_caption(
    image,
    cue: CaptionCue,
    t: float,
    style: CaptionStyle,
    video_width: int,
    video_height: int,
    font_for,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    font_size = max(1, round(style.font_size))
    stroke_width = _caption_stroke_width(style)
    box_width = round(_clamp(style.max_width, MIN_CAPTION_MAX_WIDTH, MAX_CAPTION_MAX_WIDTH) * video_width)
    x_center = _clamp_caption_x(style.x, style.max_width) * video_width
    box_left = round(_clamp(x_center - box_width / 2, PREVIEW_EDGE_INSET * video_width, video_width - box_width - PREVIEW_EDGE_INSET * video_width))
    font = font_for(font_size)
    font_metrics = _font_line_metrics(draw, font)
    line_height = max(1, round(max(font_size * CAPTION_LINE_HEIGHT, font_metrics.height)))
    space_width = max(1, draw.textlength(" ", font=font))
    words = _layout_words(draw, cue, t, style, font_size, font_for)
    if not words:
        return

    lines = _wrap_words(words, box_width, space_width, line_height)
    block_height = line_height * len(lines)
    y = round(_clamp(style.y * video_height - block_height / 2, 0, video_height - block_height))
    base_color = _hex_rgba(style.base_color)
    highlight_color = _hex_rgba(style.highlight_color)
    outline_color = _hex_rgba(style.outline_color)
    shadow_color = _hex_rgba(style.shadow_color, round(_clamp(style.shadow_opacity, 0.0, 1.0) * 255))
    shadow_blur = max(0.0, style.shadow_blur)
    shadow_offset = max(0.0, style.shadow_offset)

    for line_words, line_width, line_height in lines:
        if style.align == "left":
            x = box_left
        elif style.align == "right":
            x = box_left + box_width - line_width
        else:
            x = box_left + (box_width - line_width) / 2

        baseline = round(y + (line_height - font_metrics.height) / 2 + font_metrics.ascent)

        for index, item in enumerate(line_words):
            if (style.highlight_mode or "progressive") == "progressive":
                _draw_text_shadow(
                    image,
                    item.text,
                    item.font,
                    x,
                    baseline,
                    shadow_color,
                    shadow_blur,
                    shadow_offset,
                    stroke_width,
                )
                _draw_text(image, item.text, item.font, x, baseline, base_color, outline_color, stroke_width)
                if item.fill_pct > 0:
                    _draw_text_clipped(
                        image,
                        item.text,
                        item.font,
                        x,
                        baseline,
                        highlight_color,
                        outline_color,
                        0,
                        item.fill_pct,
                        item.width,
                    )
            else:
                fill = highlight_color if item.is_active else base_color
                draw_font = item.font
                draw_x = x
                draw_baseline = baseline
                if item.pop_scale != 1.0:
                    draw_font = font_for(round(font_size * item.pop_scale))
                    scaled_width = max(1.0, draw.textlength(item.text, font=draw_font))
                    scaled_bbox = draw.textbbox((0, 0), item.text, font=draw_font, anchor="ls", stroke_width=0)
                    scaled_descent = max(0.0, float(scaled_bbox[3]))
                    draw_x = x - (scaled_width - item.width) / 2
                    draw_baseline = baseline + item.descent - scaled_descent
                _draw_text_shadow(
                    image,
                    item.text,
                    draw_font,
                    draw_x,
                    draw_baseline,
                    shadow_color,
                    shadow_blur,
                    shadow_offset,
                    stroke_width,
                )
                _draw_text(image, item.text, draw_font, draw_x, draw_baseline, fill, outline_color, stroke_width)

            x += item.width
            if index < len(line_words) - 1:
                x += space_width
        y += line_height


def _layout_words(draw, cue: CaptionCue, t: float, style: CaptionStyle, font_size: int, font_for) -> list[WordLayout]:
    items: list[WordLayout] = []
    highlight_mode = style.highlight_mode or "progressive"
    active_intervals = word_highlight_intervals(cue) if highlight_mode in {"active_word", "pop_word"} else {}
    font = font_for(font_size)
    for word in cue.words:
        text = word.text.strip()
        if not text:
            continue
        active_interval = active_intervals.get(word.id)
        if active_interval:
            is_active = active_interval.start <= t < active_interval.end
            active_start = active_interval.start
        else:
            is_active = word.start <= t < word.end
            active_start = word.start
        bbox = draw.textbbox((0, 0), text, font=font, anchor="ls", stroke_width=0)
        items.append(WordLayout(
            word=word,
            text=text,
            width=max(1.0, draw.textlength(text, font=font)),
            descent=max(0.0, float(bbox[3])),
            font=font,
            fill_pct=_word_progress(word, t) if highlight_mode == "progressive" else (1.0 if is_active else 0.0),
            is_active=is_active,
            active_start=active_start,
            pop_scale=_pop_scale(active_start, t) if highlight_mode == "pop_word" and is_active else 1.0,
        ))
    return items


def _wrap_words(words: list[WordLayout], box_width: int, space_width: float, line_height: int) -> list[tuple[list[WordLayout], float, int]]:
    lines: list[tuple[list[WordLayout], float, int]] = []
    current: list[WordLayout] = []
    current_width = 0.0

    def flush() -> None:
        nonlocal current, current_width
        if not current:
            return
        lines.append((current, current_width, line_height))
        current = []
        current_width = 0.0

    for item in words:
        next_width = item.width if not current else current_width + space_width + item.width
        if current and next_width > box_width:
            flush()
        current_width = item.width if not current else current_width + space_width + item.width
        current.append(item)
    flush()
    return lines


def _draw_text(image, text: str, font, x: float, baseline: float, fill, outline, stroke_width: int) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    draw.text(
        (round(x), round(baseline)),
        text,
        font=font,
        anchor="ls",
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=outline,
    )


def _draw_text_shadow(
    image,
    text: str,
    font,
    x: float,
    baseline: float,
    fill,
    blur: float,
    offset: float,
    stroke_width: int,
) -> None:
    if fill[3] <= 0 or (blur <= 0 and offset <= 0):
        return

    from PIL import Image, ImageDraw, ImageFilter

    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font, anchor="ls", stroke_width=stroke_width)
    pad = max(2, math.ceil(blur * 3 + stroke_width + offset + 4))
    width = max(1, math.ceil(bbox[2] - bbox[0] + pad * 2))
    height = max(1, math.ceil(bbox[3] - bbox[1] + pad * 2))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text(
        (pad - bbox[0], pad - bbox[1] + offset),
        text,
        font=font,
        anchor="ls",
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=fill,
    )
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(radius=blur))
    image.alpha_composite(layer, (round(x + bbox[0] - pad), round(baseline + bbox[1] - pad)))


def _draw_text_clipped(
    image,
    text: str,
    font,
    x: float,
    baseline: float,
    fill,
    outline,
    stroke_width: int,
    fill_pct: float,
    layout_width: float,
) -> None:
    from PIL import Image, ImageDraw

    fill_pct = _clamp(fill_pct, 0.0, 1.0)
    if fill_pct <= 0:
        return
    draw = ImageDraw.Draw(image)
    bbox = draw.textbbox((0, 0), text, font=font, anchor="ls", stroke_width=0)
    pad = stroke_width + 2
    width = max(1, math.ceil(layout_width + pad * 2))
    height = max(1, math.ceil(bbox[3] - bbox[1] + pad * 2))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text(
        (pad - bbox[0], pad - bbox[1]),
        text,
        font=font,
        anchor="ls",
        fill=fill,
        stroke_width=stroke_width,
        stroke_fill=outline,
    )
    crop_width = max(1, min(width, pad + round(layout_width * fill_pct)))
    image.alpha_composite(
        layer.crop((0, 0, crop_width, height)),
        (round(x + bbox[0] - pad), round(baseline + bbox[1] - pad)),
    )


def _font_line_metrics(draw, font) -> FontLineMetrics:
    bbox = draw.textbbox((0, 0), "Hgyp", font=font, anchor="ls", stroke_width=0)
    ascent = max(1.0, -float(bbox[1]))
    descent = max(0.0, float(bbox[3]))
    return FontLineMetrics(ascent=ascent, descent=descent, height=ascent + descent)


def _caption_font_path() -> str:
    env_path = os.environ.get("CAPTION_FONT_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    bundled = os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "Poppins-ExtraBold.ttf")
    candidates: list[str] = [os.path.abspath(bundled)]
    for base in ("~/Library/Fonts", "/Library/Fonts", "/System/Library/Fonts", "/System/Library/Fonts/Supplemental"):
        root = os.path.expanduser(base)
        candidates.extend(glob.glob(os.path.join(root, "Poppins*ExtraBold*.ttf")))
        candidates.extend(glob.glob(os.path.join(root, "Poppins*Bold*.ttf")))
        candidates.extend(glob.glob(os.path.join(root, "Poppins*.ttf")))

    candidates.extend([
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    raise RuntimeError("No usable caption font found. Set CAPTION_FONT_PATH to a .ttf or .ttc font file.")


def _caption_stroke_width(style: CaptionStyle) -> int:
    return max(0, round(style.outline_width))


def _hex_rgba(value: str, alpha: int = 255) -> tuple[int, int, int, int]:
    match = value.strip().lstrip("#")
    if len(match) != 6:
        match = "ffffff"
    try:
        return (int(match[0:2], 16), int(match[2:4], 16), int(match[4:6], 16), alpha)
    except ValueError:
        return (255, 255, 255, alpha)


def _word_progress(word: CaptionWord, t: float) -> float:
    if t <= word.start:
        return 0.0
    if t >= word.end:
        return 1.0
    return _clamp((t - word.start) / max(0.01, word.end - word.start), 0.0, 1.0)


def _pop_scale(start: float, t: float) -> float:
    elapsed = t - start
    if elapsed < 0 or elapsed >= CAPTION_POP_DURATION:
        return 1.0
    pct = elapsed / CAPTION_POP_DURATION
    if pct <= CAPTION_POP_PEAK_AT:
        return 1.0 + (CAPTION_POP_PEAK - 1.0) * (pct / CAPTION_POP_PEAK_AT)
    return CAPTION_POP_PEAK - (CAPTION_POP_PEAK - 1.0) * ((pct - CAPTION_POP_PEAK_AT) / (1.0 - CAPTION_POP_PEAK_AT))


def _clamp_caption_x(x: float, width: float) -> float:
    max_width = _clamp(width, MIN_CAPTION_MAX_WIDTH, MAX_CAPTION_MAX_WIDTH)
    half_width = max_width / 2
    return _clamp(x, PREVIEW_EDGE_INSET + half_width, 1 - PREVIEW_EDGE_INSET - half_width)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
