import type { CaptionCue, CaptionStyle, CaptionWord } from '../../types/project';

const CAPTION_LINE_HEIGHT = 1.12;
const CAPTION_POP_DURATION = 0.18;
const CAPTION_POP_PEAK = 1.18;
const CAPTION_POP_PEAK_AT = 0.45;
const DEFAULT_CAPTION_MAX_WIDTH = 0.72;
const MIN_CAPTION_MAX_WIDTH = 0.12;
const MAX_CAPTION_MAX_WIDTH = 0.96;
const PREVIEW_EDGE_INSET = 0.02;
const DEFAULT_OUTLINE_WIDTH = 4;
const DEFAULT_SHADOW_COLOR = '#000000';
const DEFAULT_SHADOW_OPACITY = 0.82;
const DEFAULT_SHADOW_BLUR = 6;
const DEFAULT_SHADOW_OFFSET = 3;
const MIN_ACTIVE_WORD_DURATION = 0.18;

interface VideoSize {
  width: number;
  height: number;
}

interface TextMetricsBox {
  width: number;
  height: number;
  ascent: number;
  descent: number;
}

interface WordLayout {
  word: CaptionWord;
  text: string;
  width: number;
  descent: number;
  fillPct: number;
  isActive: boolean;
  activeStart: number;
  popScale: number;
}

interface CaptionLine {
  words: WordLayout[];
  width: number;
  height: number;
}

export interface CaptionCanvasResult {
  blockHeight: number;
}

export function drawCaptionCanvas(
  canvas: HTMLCanvasElement,
  cue: CaptionCue | null,
  t: number,
  style: CaptionStyle,
  videoSize: VideoSize,
): CaptionCanvasResult | null {
  if (canvas.width !== videoSize.width) canvas.width = videoSize.width;
  if (canvas.height !== videoSize.height) canvas.height = videoSize.height;

  const ctx = canvas.getContext('2d');
  if (!ctx) return null;

  ctx.clearRect(0, 0, videoSize.width, videoSize.height);
  if (!cue) return null;

  return drawCaption(ctx, cue, t, style, videoSize.width, videoSize.height);
}

export function fallbackCaptionBlockHeight(style: CaptionStyle): number {
  return Math.max(1, Math.round(getFontSize(style) * CAPTION_LINE_HEIGHT));
}

export function getWordHighlightIntervals(cue: CaptionCue): Map<string, { start: number; end: number }> {
  const words = cue.words.filter((word) => word.text.trim().length > 0);
  const intervals = new Map<string, { start: number; end: number }>();
  if (words.length === 0) return intervals;

  const cueStart = finiteNumber(cue.start, words[0].start);
  let cueEnd = finiteNumber(cue.end, words[words.length - 1].end);
  if (cueEnd <= cueStart) {
    cueEnd = cueStart + Math.max(MIN_ACTIVE_WORD_DURATION, words.length * MIN_ACTIVE_WORD_DURATION);
  }

  if (!needsWordTimingRepair(words)) {
    for (const word of words) {
      intervals.set(word.id, {
        start: finiteNumber(word.start, cueStart),
        end: finiteNumber(word.end, cueEnd),
      });
    }
    return intervals;
  }

  const cueDuration = Math.max(0.001, cueEnd - cueStart);
  const weights = words.map((word) => (
    Math.max(
      MIN_ACTIVE_WORD_DURATION,
      finiteNumber(word.end, cueStart) - finiteNumber(word.start, cueStart),
    )
  ));
  const totalWeight = weights.reduce((sum, value) => sum + value, 0) || words.length;
  let cursor = cueStart;

  words.forEach((word, index) => {
    const end = index === words.length - 1
      ? cueEnd
      : Math.min(cueEnd, cursor + Math.max(0.001, cueDuration * (weights[index] / totalWeight)));
    intervals.set(word.id, { start: cursor, end });
    cursor = end;
  });

  return intervals;
}

function drawCaption(
  ctx: CanvasRenderingContext2D,
  cue: CaptionCue,
  t: number,
  style: CaptionStyle,
  videoWidth: number,
  videoHeight: number,
): CaptionCanvasResult | null {
  const fontSize = getFontSize(style);
  const strokeWidth = getOutlineWidth(style);
  const maxWidth = clamp(getStyleNumber(style.max_width, DEFAULT_CAPTION_MAX_WIDTH), MIN_CAPTION_MAX_WIDTH, MAX_CAPTION_MAX_WIDTH);
  const boxWidth = Math.round(maxWidth * videoWidth);
  const xCenter = clampCaptionX(style.x, maxWidth) * videoWidth;
  const edgeInset = PREVIEW_EDGE_INSET * videoWidth;
  const boxLeft = Math.round(clamp(xCenter - boxWidth / 2, edgeInset, videoWidth - boxWidth - edgeInset));
  const fontMetrics = measureFontMetrics(ctx, fontSize);
  const lineHeight = Math.max(1, Math.round(Math.max(fontSize * CAPTION_LINE_HEIGHT, fontMetrics.height)));
  const spaceWidth = Math.max(1, measureText(ctx, ' ', fontSize).width);
  const words = layoutWords(ctx, cue, t, style, fontSize);
  if (words.length === 0) return null;

  const lines = wrapWords(words, boxWidth, spaceWidth, lineHeight);
  const blockHeight = lineHeight * lines.length;
  let y = Math.round(clamp(style.y * videoHeight - blockHeight / 2, 0, videoHeight - blockHeight));

  const baseColor = getStyleColor(style.base_color, '#ffffff');
  const highlightColor = getStyleColor(style.highlight_color, '#ffd34d');
  const outlineColor = getStyleColor(style.outline_color, '#000000');
  const shadowColor = rgbaString(
    getStyleColor(style.shadow_color, DEFAULT_SHADOW_COLOR),
    clamp(getStyleNumber(style.shadow_opacity, DEFAULT_SHADOW_OPACITY), 0, 1),
  );
  const shadowBlur = Math.max(0, getStyleNumber(style.shadow_blur, DEFAULT_SHADOW_BLUR));
  const shadowOffset = Math.max(0, getStyleNumber(style.shadow_offset, DEFAULT_SHADOW_OFFSET));

  for (const line of lines) {
    let x: number;
    if (style.align === 'left') {
      x = boxLeft;
    } else if (style.align === 'right') {
      x = boxLeft + boxWidth - line.width;
    } else {
      x = boxLeft + (boxWidth - line.width) / 2;
    }

    const baseline = Math.round(y + (line.height - fontMetrics.height) / 2 + fontMetrics.ascent);
    const lineTop = y;

    for (const [index, item] of line.words.entries()) {
      const highlightMode = style.highlight_mode ?? 'progressive';

      if (highlightMode === 'progressive') {
        drawTextShadow(ctx, item.text, fontSize, x, baseline, shadowColor, shadowBlur, shadowOffset, strokeWidth);
        drawText(ctx, item.text, fontSize, x, baseline, baseColor, outlineColor, strokeWidth);
        if (item.fillPct > 0) {
          drawTextClipped(ctx, item.text, fontSize, x, baseline, lineTop, highlightColor, item.fillPct, item.width, line.height);
        }
      } else {
        const fill = item.isActive ? highlightColor : baseColor;
        let drawFontSize = fontSize;
        let drawX = x;
        let drawBaseline = baseline;

        if (item.popScale !== 1) {
          drawFontSize = Math.round(fontSize * item.popScale);
          const scaledMetrics = measureText(ctx, item.text, drawFontSize);
          drawX = x - (scaledMetrics.width - item.width) / 2;
          drawBaseline = baseline + item.descent - scaledMetrics.descent;
        }

        drawTextShadow(ctx, item.text, drawFontSize, drawX, drawBaseline, shadowColor, shadowBlur, shadowOffset, strokeWidth);
        drawText(ctx, item.text, drawFontSize, drawX, drawBaseline, fill, outlineColor, strokeWidth);
      }

      x += item.width;
      if (index < line.words.length - 1) x += spaceWidth;
    }

    y += line.height;
  }

  return { blockHeight };
}

function layoutWords(
  ctx: CanvasRenderingContext2D,
  cue: CaptionCue,
  t: number,
  style: CaptionStyle,
  fontSize: number,
): WordLayout[] {
  const items: WordLayout[] = [];
  const highlightMode = style.highlight_mode ?? 'progressive';
  const activeIntervals = highlightMode === 'progressive' ? null : getWordHighlightIntervals(cue);

  for (const word of cue.words) {
    const text = word.text.trim();
    if (!text) continue;

    const activeInterval = activeIntervals?.get(word.id);
    const isActive = activeInterval
      ? activeInterval.start <= t && t < activeInterval.end
      : word.start <= t && t < word.end;
    const activeStart = activeInterval?.start ?? word.start;
    const metrics = measureText(ctx, text, fontSize);

    items.push({
      word,
      text,
      width: Math.max(1, metrics.width),
      descent: metrics.descent,
      fillPct: highlightMode === 'progressive' ? wordProgress(word.start, word.end, t) : (isActive ? 1 : 0),
      isActive,
      activeStart,
      popScale: highlightMode === 'pop_word' && isActive ? popScale(activeStart, t) : 1,
    });
  }

  return items;
}

function wrapWords(words: WordLayout[], boxWidth: number, spaceWidth: number, lineHeight: number): CaptionLine[] {
  const lines: CaptionLine[] = [];
  let current: WordLayout[] = [];
  let currentWidth = 0;

  function flush() {
    if (current.length === 0) return;
    lines.push({ words: current, width: currentWidth, height: lineHeight });
    current = [];
    currentWidth = 0;
  }

  for (const item of words) {
    const nextWidth = current.length === 0 ? item.width : currentWidth + spaceWidth + item.width;
    if (current.length > 0 && nextWidth > boxWidth) flush();
    currentWidth = current.length === 0 ? item.width : currentWidth + spaceWidth + item.width;
    current.push(item);
  }

  flush();
  return lines;
}

function drawTextShadow(
  ctx: CanvasRenderingContext2D,
  text: string,
  fontSize: number,
  x: number,
  baseline: number,
  fill: string,
  blur: number,
  offset: number,
  strokeWidth: number,
) {
  if ((blur <= 0 && offset <= 0) || fill.endsWith(', 0)')) return;
  ctx.save();
  if (blur > 0) ctx.filter = `blur(${blur}px)`;
  drawText(ctx, text, fontSize, x, baseline + offset, fill, fill, strokeWidth);
  ctx.restore();
}

function drawText(
  ctx: CanvasRenderingContext2D,
  text: string,
  fontSize: number,
  x: number,
  baseline: number,
  fill: string,
  outline: string,
  strokeWidth: number,
) {
  const roundedX = Math.round(x);
  const roundedBaseline = Math.round(baseline);

  ctx.save();
  ctx.font = captionFont(fontSize);
  ctx.textBaseline = 'alphabetic';
  ctx.textAlign = 'left';
  ctx.lineJoin = 'round';
  if (strokeWidth > 0) {
    ctx.lineWidth = strokeWidth * 2;
    ctx.strokeStyle = outline;
    ctx.strokeText(text, roundedX, roundedBaseline);
  }
  ctx.fillStyle = fill;
  ctx.fillText(text, roundedX, roundedBaseline);
  ctx.restore();
}

function drawTextClipped(
  ctx: CanvasRenderingContext2D,
  text: string,
  fontSize: number,
  x: number,
  baseline: number,
  lineTop: number,
  fill: string,
  fillPct: number,
  layoutWidth: number,
  lineHeight: number,
) {
  const clippedWidth = Math.max(0, Math.min(layoutWidth, layoutWidth * clamp(fillPct, 0, 1)));
  if (clippedWidth <= 0) return;

  ctx.save();
  ctx.beginPath();
  ctx.rect(x, lineTop - 2, clippedWidth, lineHeight + 4);
  ctx.clip();
  drawText(ctx, text, fontSize, x, baseline, fill, fill, 0);
  ctx.restore();
}

function measureText(ctx: CanvasRenderingContext2D, text: string, fontSize: number): TextMetricsBox {
  ctx.save();
  ctx.font = captionFont(fontSize);
  const metrics = ctx.measureText(text);
  ctx.restore();

  const ascent = metrics.actualBoundingBoxAscent || fontSize * 0.8;
  const descent = metrics.actualBoundingBoxDescent || fontSize * 0.2;
  return {
    width: metrics.width,
    height: ascent + descent,
    ascent,
    descent,
  };
}

function measureFontMetrics(ctx: CanvasRenderingContext2D, fontSize: number): TextMetricsBox {
  return measureText(ctx, 'Hgyp', fontSize);
}

function captionFont(fontSize: number): string {
  return `800 ${Math.max(1, Math.round(fontSize))}px "Caption Poppins", sans-serif`;
}

function getFontSize(style: CaptionStyle): number {
  return Math.max(1, Math.round(getStyleNumber(style.font_size, 48)));
}

function getOutlineWidth(style: CaptionStyle): number {
  return Math.max(0, Math.round(getStyleNumber(style.outline_width, DEFAULT_OUTLINE_WIDTH)));
}

function wordProgress(start: number, end: number, t: number): number {
  if (t <= start) return 0;
  if (t >= end) return 1;
  return clamp((t - start) / Math.max(0.01, end - start), 0, 1);
}

function popScale(start: number, t: number): number {
  const elapsed = t - start;
  if (elapsed < 0 || elapsed >= CAPTION_POP_DURATION) return 1;
  const pct = elapsed / CAPTION_POP_DURATION;
  if (pct <= CAPTION_POP_PEAK_AT) {
    return 1 + (CAPTION_POP_PEAK - 1) * (pct / CAPTION_POP_PEAK_AT);
  }
  return CAPTION_POP_PEAK - (CAPTION_POP_PEAK - 1) * ((pct - CAPTION_POP_PEAK_AT) / (1 - CAPTION_POP_PEAK_AT));
}

function clampCaptionX(x: number, width: number): number {
  const maxWidth = clamp(width, MIN_CAPTION_MAX_WIDTH, MAX_CAPTION_MAX_WIDTH);
  const halfWidth = maxWidth / 2;
  return clamp(x, PREVIEW_EDGE_INSET + halfWidth, 1 - PREVIEW_EDGE_INSET - halfWidth);
}

function needsWordTimingRepair(words: CaptionWord[]): boolean {
  let previousStart: number | null = null;
  for (const word of words) {
    const start = finiteNumber(word.start, 0);
    const end = finiteNumber(word.end, start);
    if (end - start < MIN_ACTIVE_WORD_DURATION) return true;
    if (previousStart !== null && start <= previousStart) return true;
    previousStart = start;
  }
  return false;
}

function rgbaString(hex: string, opacity: number): string {
  const rgb = hexToRgb(hex);
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
}

function hexToRgb(value: string): { r: number; g: number; b: number } {
  const hex = value.replace('#', '');
  return {
    r: Number.parseInt(hex.slice(0, 2), 16),
    g: Number.parseInt(hex.slice(2, 4), 16),
    b: Number.parseInt(hex.slice(4, 6), 16),
  };
}

function getStyleNumber(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function getStyleColor(value: string | undefined, fallback: string): string {
  return typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value) ? value : fallback;
}

function finiteNumber(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
