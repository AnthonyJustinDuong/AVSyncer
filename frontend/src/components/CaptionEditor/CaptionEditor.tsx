import { Combine, Download, Play, Save, Scissors, Split, Type } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import type { AppStage, CaptionCue, CaptionProject, CaptionStyle } from '../../types/project';
import {
  drawCaptionCanvas,
  fallbackCaptionBlockHeight,
} from './captionCanvasRenderer';
import './CaptionEditor.css';

interface Props {
  project: CaptionProject;
  stage: AppStage;
  downloadUrl: string | null;
  exportProgress: number;
  onProjectChange: (project: CaptionProject) => void;
  onSave: () => void;
  onExport: () => void;
}

const HIGHLIGHT_OPTIONS: Array<{
  mode: CaptionStyle['highlight_mode'];
  label: string;
  description: string;
}> = [
  {
    mode: 'progressive',
    label: 'Sweep Fill',
    description: 'Fills each word from left to right as it is spoken.',
  },
  {
    mode: 'active_word',
    label: 'Whole Word',
    description: 'Highlights the whole word instantly during its timing.',
  },
  {
    mode: 'pop_word',
    label: 'Pop Word',
    description: 'Highlights the whole word and pops it on start.',
  },
];

const DEFAULT_CAPTION_MAX_WIDTH = 0.72;
const MIN_CAPTION_MAX_WIDTH = 0.12;
const MAX_CAPTION_MAX_WIDTH = 0.96;
const PREVIEW_EDGE_INSET = 0.02;
const DEFAULT_OUTLINE_WIDTH = 4;
const DEFAULT_BASE_COLOR = '#ffffff';
const DEFAULT_HIGHLIGHT_COLOR = '#ffd34d';
const DEFAULT_OUTLINE_COLOR = '#000000';
const DEFAULT_SHADOW_COLOR = '#000000';
const DEFAULT_SHADOW_OPACITY = 0.82;
const DEFAULT_SHADOW_BLUR = 6;
const DEFAULT_SHADOW_OFFSET = 3;

interface PreviewFrame {
  left: number;
  top: number;
  width: number;
  height: number;
}

export default function CaptionEditor({
  project,
  stage,
  downloadUrl,
  exportProgress,
  onProjectChange,
  onSave,
  onExport,
}: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const previewRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [resizingWidth, setResizingWidth] = useState(false);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number } | null>(null);
  const [previewFrame, setPreviewFrame] = useState<PreviewFrame | null>(null);
  const [captionFontReady, setCaptionFontReady] = useState(false);
  const [captionBlockHeight, setCaptionBlockHeight] = useState(() => fallbackCaptionBlockHeight(project.style));
  const [selectedWord, setSelectedWord] = useState<{ cueId: string; wordId: string } | null>(null);
  const isExporting = stage === 'exporting';
  const isDone = stage === 'done';
  const highlightMode = project.style.highlight_mode ?? 'progressive';
  const captionMaxWidth = getCaptionMaxWidth(project.style);
  const captionX = clampCaptionX(project.style.x, captionMaxWidth);
  const outlineWidth = clamp(getStyleNumber(project.style.outline_width, DEFAULT_OUTLINE_WIDTH), 0, 12);
  const shadowOpacity = clamp(getStyleNumber(project.style.shadow_opacity, DEFAULT_SHADOW_OPACITY), 0, 1);
  const shadowBlur = clamp(getStyleNumber(project.style.shadow_blur, DEFAULT_SHADOW_BLUR), 0, 24);
  const shadowOffset = clamp(getStyleNumber(project.style.shadow_offset, DEFAULT_SHADOW_OFFSET), 0, 24);
  const previewCaptionStyle = useMemo<CaptionStyle>(() => ({
    ...project.style,
    base_color: getStyleColor(project.style.base_color, DEFAULT_BASE_COLOR),
    highlight_color: getStyleColor(project.style.highlight_color, DEFAULT_HIGHLIGHT_COLOR),
    outline_color: getStyleColor(project.style.outline_color, DEFAULT_OUTLINE_COLOR),
    shadow_color: getStyleColor(project.style.shadow_color, DEFAULT_SHADOW_COLOR),
  }), [project.style]);
  const previewScale = previewFrame && videoSize ? Math.max(0.01, previewFrame.width / videoSize.width) : 1;
  const overlayBoxHeight = Math.max(18, captionBlockHeight * previewScale);
  const selection = useMemo(() => {
    if (!selectedWord) return null;
    const cueIndex = project.cues.findIndex((cue) => cue.id === selectedWord.cueId);
    const cue = project.cues[cueIndex];
    if (!cue) return null;
    const wordIndex = cue.words.findIndex((word) => word.id === selectedWord.wordId);
    if (wordIndex < 0) return null;
    return { cue, cueIndex, wordIndex, word: cue.words[wordIndex] };
  }, [project.cues, selectedWord]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let raf = 0;
    function tick() {
      if (video) setCurrentTime(video.currentTime);
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  const activeCue = useMemo(
    () => project.cues.find((cue) => currentTime >= cue.start && currentTime <= cue.end) ?? null,
    [project.cues, currentTime],
  );

  useEffect(() => {
    let cancelled = false;
    if (!('fonts' in document)) {
      setCaptionFontReady(true);
      return;
    }

    document.fonts.load('800 48px "Caption Poppins"')
      .then(() => {
        if (!cancelled) setCaptionFontReady(true);
      })
      .catch(() => {
        if (!cancelled) setCaptionFontReady(true);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setVideoSize(null);
    setPreviewFrame(null);
  }, [project.video_url]);

  useEffect(() => {
    const node = previewRef.current;
    if (node === null) return;

    const updatePreviewFrame = () => {
      const rect = node.getBoundingClientRect();
      setPreviewFrame((current) => {
        const next = getContainedVideoFrame(rect.width, rect.height, videoSize);
        if (
          current &&
          Math.abs(current.left - next.left) < 0.5 &&
          Math.abs(current.top - next.top) < 0.5 &&
          Math.abs(current.width - next.width) < 0.5 &&
          Math.abs(current.height - next.height) < 0.5
        ) {
          return current;
        }
        return next;
      });
    };

    updatePreviewFrame();
    const observer = new ResizeObserver(updatePreviewFrame);
    observer.observe(node);
    return () => observer.disconnect();
  }, [project.video_url, videoSize?.width, videoSize?.height]);

  useEffect(() => {
    if (!videoSize || !captionFontReady) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const result = drawCaptionCanvas(canvas, activeCue, currentTime, previewCaptionStyle, videoSize);
    const nextHeight = result?.blockHeight ?? fallbackCaptionBlockHeight(previewCaptionStyle);
    setCaptionBlockHeight((current) => (
      Math.abs(current - nextHeight) > 0.5 ? nextHeight : current
    ));
  }, [activeCue, captionFontReady, currentTime, previewCaptionStyle, videoSize]);

  function updateStyle(patch: Partial<CaptionStyle>) {
    onProjectChange({ ...project, style: { ...project.style, ...patch } });
  }

  function updateCaptionMaxWidth(nextWidth: number) {
    const max_width = clampCaptionWidth(nextWidth);
    updateStyle({
      max_width,
      x: clampCaptionX(project.style.x, max_width),
    });
  }

  function updateCue(nextCue: CaptionCue) {
    onProjectChange({
      ...project,
      cues: project.cues.map((cue) => (cue.id === nextCue.id ? nextCue : cue)),
    });
  }

  function updateWord(cue: CaptionCue, wordId: string, text: string) {
    updateCue({
      ...cue,
      words: cue.words.map((word) => (word.id === wordId ? { ...word, text } : word)),
    });
  }

  function splitCueAtSelection() {
    if (!selection) return;
    const { cue, cueIndex, wordIndex } = selection;
    if (wordIndex <= 0 || wordIndex >= cue.words.length) return;
    const left = cue.words.slice(0, wordIndex);
    const right = cue.words.slice(wordIndex);
    const rightCue = cueFromWords(right);
    const nextCues = [
      ...project.cues.slice(0, cueIndex),
      cueFromWords(left),
      rightCue,
      ...project.cues.slice(cueIndex + 1),
    ];
    onProjectChange({ ...project, cues: nextCues });
    setSelectedWord({ cueId: rightCue.id, wordId: right[0].id });
  }

  function mergeSelectionUp() {
    if (!selection || selection.cueIndex <= 0) return;
    const prev = project.cues[selection.cueIndex - 1];
    const current = selection.cue;
    const joined = cueFromWords([...prev.words, ...current.words]);
    const nextCues = [
      ...project.cues.slice(0, selection.cueIndex - 1),
      joined,
      ...project.cues.slice(selection.cueIndex + 1),
    ];
    onProjectChange({ ...project, cues: nextCues });
    setSelectedWord({ cueId: joined.id, wordId: selection.word.id });
  }

  function mergeSelectionDown() {
    if (!selection || selection.cueIndex >= project.cues.length - 1) return;
    const current = selection.cue;
    const next = project.cues[selection.cueIndex + 1];
    const joined = cueFromWords([...current.words, ...next.words]);
    const nextCues = [
      ...project.cues.slice(0, selection.cueIndex),
      joined,
      ...project.cues.slice(selection.cueIndex + 2),
    ];
    onProjectChange({ ...project, cues: nextCues });
    setSelectedWord({ cueId: joined.id, wordId: selection.word.id });
  }

  function seekTo(t: number) {
    if (!videoRef.current) return;
    videoRef.current.currentTime = Math.max(0, Math.min(project.duration, t));
    videoRef.current.play().catch(() => { /* ignore autoplay rejection */ });
  }

  function handleOverlayPointerDown(e: React.PointerEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(true);
    e.currentTarget.setPointerCapture(e.pointerId);
    moveOverlay(e.clientX, e.clientY);
  }

  function handleOverlayPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragging) return;
    moveOverlay(e.clientX, e.clientY);
  }

  function handleOverlayPointerEnd(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragging) return;
    setDragging(false);
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  }

  function moveOverlay(clientX: number, clientY: number) {
    const point = getVideoPoint(clientX, clientY);
    if (!point) return;
    updateStyle({
      x: clampCaptionX(point.x, captionMaxWidth),
      y: clamp(point.y, PREVIEW_EDGE_INSET, 1 - PREVIEW_EDGE_INSET),
    });
  }

  function handleWidthPointerDown(e: React.PointerEvent<HTMLButtonElement>) {
    e.preventDefault();
    e.stopPropagation();
    setResizingWidth(true);
    e.currentTarget.setPointerCapture(e.pointerId);
    resizeOverlayWidth(e.clientX);
  }

  function handleWidthPointerMove(e: React.PointerEvent<HTMLButtonElement>) {
    if (!resizingWidth) return;
    resizeOverlayWidth(e.clientX);
  }

  function handleWidthPointerEnd(e: React.PointerEvent<HTMLButtonElement>) {
    if (!resizingWidth) return;
    setResizingWidth(false);
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* already released */ }
  }

  function resizeOverlayWidth(clientX: number) {
    const point = getVideoPoint(clientX, 0);
    if (!point) return;
    const boxLeft = clamp(
      captionX - captionMaxWidth / 2,
      PREVIEW_EDGE_INSET,
      1 - PREVIEW_EDGE_INSET - MIN_CAPTION_MAX_WIDTH,
    );
    const pointerX = clamp(
      point.x,
      boxLeft + MIN_CAPTION_MAX_WIDTH,
      1 - PREVIEW_EDGE_INSET,
    );
    const max_width = clampCaptionWidth(pointerX - boxLeft);
    updateStyle({
      max_width,
      x: clampCaptionX(boxLeft + max_width / 2, max_width),
    });
  }

  function handleWidthKeyDown(e: React.KeyboardEvent<HTMLButtonElement>) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    const delta = e.key === 'ArrowLeft' ? -0.02 : 0.02;
    updateCaptionMaxWidth(captionMaxWidth + delta);
  }

  function getVideoPoint(clientX: number, clientY: number): { x: number; y: number } | null {
    const container = previewRef.current;
    if (!container) return null;
    const rect = container.getBoundingClientRect();
    const frame = getContainedVideoFrame(rect.width, rect.height, videoSize);
    if (frame.width <= 0 || frame.height <= 0) return null;
    return {
      x: clamp((clientX - rect.left - frame.left) / frame.width, 0, 1),
      y: clamp((clientY - rect.top - frame.top) / frame.height, 0, 1),
    };
  }

  return (
    <div className="caption-editor">
      <div className="caption-left">
        <div
          className="caption-preview"
          ref={previewRef}
          style={videoSize ? { aspectRatio: `${videoSize.width} / ${videoSize.height}` } : undefined}
        >
          <video
            ref={videoRef}
            src={project.video_url}
            controls
            onLoadedMetadata={(e) => {
              const video = e.currentTarget;
              if (video.videoWidth > 0 && video.videoHeight > 0) {
                setVideoSize({ width: video.videoWidth, height: video.videoHeight });
              }
            }}
            onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
          />
          {videoSize && previewFrame && (
            <canvas
              ref={canvasRef}
              className="caption-preview-canvas"
              width={videoSize.width}
              height={videoSize.height}
              style={{
                left: `${previewFrame.left}px`,
                top: `${previewFrame.top}px`,
                width: `${previewFrame.width}px`,
                height: `${previewFrame.height}px`,
              }}
            />
          )}
          <div
            className={[
              'caption-overlay',
              dragging ? 'dragging' : '',
              resizingWidth ? 'resizing-width' : '',
            ].filter(Boolean).join(' ')}
            style={{
              display: previewFrame ? undefined : 'none',
              left: previewFrame ? `${previewFrame.left + captionX * previewFrame.width}px` : 0,
              top: previewFrame ? `${previewFrame.top + project.style.y * previewFrame.height}px` : 0,
              width: previewFrame ? `${captionMaxWidth * previewFrame.width}px` : 0,
              height: `${overlayBoxHeight}px`,
              '--caption-preview-scale': previewScale,
            } as CSSProperties}
            onPointerDown={handleOverlayPointerDown}
            onPointerMove={handleOverlayPointerMove}
            onPointerUp={handleOverlayPointerEnd}
            onPointerCancel={handleOverlayPointerEnd}
            title="Drag caption position"
          >
            <button
              type="button"
              className="caption-width-tab"
              aria-label="Adjust caption max width"
              title="Drag to adjust caption max width"
              onPointerDown={handleWidthPointerDown}
              onPointerMove={handleWidthPointerMove}
              onPointerUp={handleWidthPointerEnd}
              onPointerCancel={handleWidthPointerEnd}
              onKeyDown={handleWidthKeyDown}
            />
          </div>
        </div>

        <div className="caption-style-panel">
          <div className="caption-style-title"><Type size={15} /> Caption Style</div>
          <label className="caption-control">
            <span>Size</span>
            <input
              type="range"
              min={8}
              max={180}
              value={project.style.font_size}
              onChange={(e) => updateStyle({ font_size: Number(e.target.value) })}
            />
            <span className="caption-control-value">{project.style.font_size}px</span>
          </label>
          <label className="caption-control">
            <span>Outline</span>
            <input
              type="range"
              min={0}
              max={12}
              step={0.5}
              value={outlineWidth}
              onChange={(e) => updateStyle({ outline_width: Number(e.target.value) })}
            />
            <span className="caption-control-value">{formatNumber(outlineWidth)}px</span>
          </label>
          <label className="caption-control">
            <span>Shadow</span>
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(shadowOpacity * 100)}
              onChange={(e) => updateStyle({ shadow_opacity: Number(e.target.value) / 100 })}
            />
            <span className="caption-control-value">{Math.round(shadowOpacity * 100)}%</span>
          </label>
          <label className="caption-control">
            <span>Blur</span>
            <input
              type="range"
              min={0}
              max={24}
              step={0.5}
              value={shadowBlur}
              onChange={(e) => updateStyle({ shadow_blur: Number(e.target.value) })}
            />
            <span className="caption-control-value">{formatNumber(shadowBlur)}px</span>
          </label>
          <label className="caption-control">
            <span>Drop</span>
            <input
              type="range"
              min={0}
              max={24}
              step={0.5}
              value={shadowOffset}
              onChange={(e) => updateStyle({ shadow_offset: Number(e.target.value) })}
            />
            <span className="caption-control-value">{formatNumber(shadowOffset)}px</span>
          </label>
          <label className="caption-control">
            <span>Width</span>
            <input
              type="range"
              min={Math.round(MIN_CAPTION_MAX_WIDTH * 100)}
              max={Math.round(MAX_CAPTION_MAX_WIDTH * 100)}
              value={Math.round(captionMaxWidth * 100)}
              onChange={(e) => updateCaptionMaxWidth(Number(e.target.value) / 100)}
            />
            <span className="caption-control-value">{Math.round(captionMaxWidth * 100)}%</span>
          </label>
          <div className="caption-color-row">
            <ColorField
              label="Text"
              value={project.style.base_color}
              fallback={DEFAULT_BASE_COLOR}
              onChange={(base_color) => updateStyle({ base_color })}
            />
            <ColorField
              label="Fill"
              value={project.style.highlight_color}
              fallback={DEFAULT_HIGHLIGHT_COLOR}
              onChange={(highlight_color) => updateStyle({ highlight_color })}
            />
            <ColorField
              label="Outline"
              value={project.style.outline_color}
              fallback={DEFAULT_OUTLINE_COLOR}
              onChange={(outline_color) => updateStyle({ outline_color })}
            />
            <ColorField
              label="Shadow"
              value={project.style.shadow_color}
              fallback={DEFAULT_SHADOW_COLOR}
              onChange={(shadow_color) => updateStyle({ shadow_color })}
            />
          </div>
          <div className="caption-highlight-panel">
            <div className="caption-panel-label">Highlight Style</div>
            <div className="caption-mode-grid" role="group" aria-label="Caption highlight style">
              {HIGHLIGHT_OPTIONS.map((option) => (
                <button
                  key={option.mode}
                  className={highlightMode === option.mode ? 'active' : ''}
                  onClick={() => updateStyle({ highlight_mode: option.mode })}
                >
                  <span>{option.label}</span>
                  <small>{option.description}</small>
                </button>
              ))}
            </div>
          </div>
          <div className="caption-align-row" role="group" aria-label="Caption alignment">
            {(['left', 'center', 'right'] as const).map((align) => (
              <button
                key={align}
                className={project.style.align === align ? 'active' : ''}
                onClick={() => updateStyle({ align })}
              >
                {align}
              </button>
            ))}
          </div>
          <div className="caption-export-row">
            <button className="btn-secondary" onClick={onSave}><Save size={15} /> Save Captions</button>
            <button className="btn-primary" disabled={isExporting || project.cues.length === 0} onClick={onExport}>
              {isExporting ? (
                <><div className="spinner-sm" /> Exporting{exportProgress > 0 ? ` ${exportProgress}%` : '...'}</>
              ) : (
                <><Scissors size={15} /> {isDone ? 'Export Again' : 'Export Captioned MP4'}</>
              )}
            </button>
            {isDone && downloadUrl && (
              <a className="btn-secondary btn-download" href={downloadUrl} download="caption_export.mp4">
                <Download size={15} /> Download MP4
              </a>
            )}
          </div>
          {isExporting && exportProgress > 0 && (
            <div className="export-progress-bar">
              <div className="export-progress-fill" style={{ width: `${exportProgress}%` }} />
            </div>
          )}
        </div>
      </div>

      <div className="caption-right">
        <div className="caption-list-header">
          <span>Caption Words</span>
          <span>{project.cues.length} cues · {formatTime(currentTime)} / {formatTime(project.duration)}</span>
        </div>
        <div className="caption-cue-toolbar">
          <button
            onClick={mergeSelectionUp}
            disabled={!selection || selection.cueIndex === 0}
          >
            <Combine size={13} />
            Merge Up
          </button>
          <button
            onClick={splitCueAtSelection}
            disabled={!selection || selection.wordIndex === 0}
          >
            <Split size={13} />
            Split
          </button>
          <button
            onClick={mergeSelectionDown}
            disabled={!selection || selection.cueIndex >= project.cues.length - 1}
          >
            <Combine size={13} />
            Merge Down
          </button>
          <span>
            {selection
              ? `Cursor: "${selection.word.text}" in Cue ${selection.cueIndex + 1}`
              : 'Click a word to place the cursor'}
          </span>
        </div>
        <div className="caption-cue-list">
          {project.cues.map((cue, cueIndex) => (
            <div key={cue.id} className={`caption-cue${activeCue?.id === cue.id ? ' active' : ''}`}>
              <button className="caption-cue-play" onClick={() => seekTo(cue.start)} aria-label="Play caption cue">
                <Play size={12} />
              </button>
              <div className="caption-cue-body">
                <div className="caption-cue-meta">
                  <span>Cue {cueIndex + 1}</span>
                  <span>{formatTime(cue.start)} - {formatTime(cue.end)}</span>
                </div>
                <div className="caption-word-grid">
                  {cue.words.map((word) => (
                    <div
                      key={word.id}
                      className={`caption-word-edit${selectedWord?.cueId === cue.id && selectedWord.wordId === word.id ? ' selected' : ''}`}
                    >
                      <input
                        value={word.text}
                        onChange={(e) => updateWord(cue, word.id, e.target.value)}
                        onClick={() => setSelectedWord({ cueId: cue.id, wordId: word.id })}
                        onFocus={() => {
                          setSelectedWord({ cueId: cue.id, wordId: word.id });
                          seekTo(word.start);
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function ColorField({
  label,
  value,
  fallback,
  onChange,
}: {
  label: string;
  value: string | undefined;
  fallback: string;
  onChange: (value: string) => void;
}) {
  const colorValue = getStyleColor(value, fallback);
  const [draft, setDraft] = useState(colorValue);

  useEffect(() => {
    setDraft(colorValue);
  }, [colorValue]);

  const isInvalid = draft.trim().length > 0 && normalizeHexColor(draft) === null;

  function handleColorChange(nextValue: string) {
    setDraft(nextValue);
    onChange(nextValue);
  }

  function handleHexChange(nextValue: string) {
    setDraft(nextValue);
    const normalized = normalizeHexColor(nextValue);
    if (normalized !== null) {
      onChange(normalized);
    }
  }

  function handleHexBlur() {
    const normalized = normalizeHexColor(draft);
    if (normalized !== null) {
      setDraft(normalized);
      onChange(normalized);
      return;
    }
    setDraft(colorValue);
  }

  return (
    <label className="caption-color-control">
      <span>{label}</span>
      <input
        className="caption-color-swatch"
        type="color"
        value={colorValue}
        onChange={(e) => handleColorChange(e.target.value)}
        aria-label={`${label} colour picker`}
      />
      <input
        className="caption-hex-input"
        type="text"
        value={draft}
        onChange={(e) => handleHexChange(e.target.value)}
        onBlur={handleHexBlur}
        placeholder="#ffffff"
        maxLength={7}
        spellCheck={false}
        aria-label={`${label} hex colour`}
        aria-invalid={isInvalid}
      />
    </label>
  );
}

function cueFromWords(words: CaptionCue['words']): CaptionCue {
  return {
    id: makeId(),
    start: words[0]?.start ?? 0,
    end: words[words.length - 1]?.end ?? words[0]?.end ?? 0,
    words,
  };
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `cue-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function getCaptionMaxWidth(style: CaptionStyle): number {
  return clampCaptionWidth(
    Number.isFinite(style.max_width) ? style.max_width : DEFAULT_CAPTION_MAX_WIDTH,
  );
}

function getContainedVideoFrame(containerWidth: number, containerHeight: number, videoSize: { width: number; height: number } | null): PreviewFrame {
  if (!videoSize || videoSize.width <= 0 || videoSize.height <= 0 || containerWidth <= 0 || containerHeight <= 0) {
    return { left: 0, top: 0, width: containerWidth, height: containerHeight };
  }

  const scale = Math.min(containerWidth / videoSize.width, containerHeight / videoSize.height);
  const width = videoSize.width * scale;
  const height = videoSize.height * scale;
  return {
    left: (containerWidth - width) / 2,
    top: (containerHeight - height) / 2,
    width,
    height,
  };
}

function getStyleNumber(value: number | undefined, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function getStyleColor(value: string | undefined, fallback: string): string {
  if (typeof value !== 'string') return fallback;
  return normalizeHexColor(value) ?? fallback;
}

function normalizeHexColor(value: string): string | null {
  const hex = value.trim().replace(/^#/, '');
  if (/^[0-9a-fA-F]{3}$/.test(hex)) {
    return `#${hex
      .split('')
      .map((character) => character + character)
      .join('')
      .toLowerCase()}`;
  }
  if (/^[0-9a-fA-F]{6}$/.test(hex)) {
    return `#${hex.toLowerCase()}`;
  }
  return null;
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

function clampCaptionWidth(width: number): number {
  return clamp(width, MIN_CAPTION_MAX_WIDTH, MAX_CAPTION_MAX_WIDTH);
}

function clampCaptionX(x: number, width: number): number {
  const halfWidth = clampCaptionWidth(width) / 2;
  return clamp(x, PREVIEW_EDGE_INSET + halfWidth, 1 - PREVIEW_EDGE_INSET - halfWidth);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 10);
  return `${m}:${String(s).padStart(2, '0')}.${ms}`;
}
