import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Play } from 'lucide-react';
import type { Partition } from '../../types/project';
import { partitionKind } from '../../types/project';
import './Timeline.css';

interface Props {
  audioUrl: string;
  duration: number;
  partitions: Partition[];
  media: HTMLMediaElement | null;
  onPartitionsChange: (parts: Partition[]) => void;
  onManualSeekIntent: () => void;
}

const STRIP_WIDTH_PX = 96;
const PX_PER_SEC_DEFAULT = 60;
const PX_PER_SEC_MIN = 5;
const PX_PER_SEC_MAX = 300;
const MIN_PARTITION_SECONDS = 0.05;
const DRAG_THRESHOLD_PX = 3;

interface DragState {
  id: string;
  edge: 'start' | 'end' | 'body';
  startY: number;
  origStart: number;
  origEnd: number;
  minStart: number;
  maxEnd: number;
  prevId: string | null;
  nextId: string | null;
  moved: boolean;
}

export default function Timeline({ duration, partitions, media, onPartitionsChange, onManualSeekIntent }: Props) {
  const laneRef = useRef<HTMLDivElement>(null);
  const stripRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<DragState | null>(null);

  const [pxPerSec, setPxPerSec] = useState(PX_PER_SEC_DEFAULT);
  const [isDragging, setIsDragging] = useState(false);
  const laneLengthPx = Math.max(1, Math.round(duration * pxPerSec));

  const videoEl = media instanceof HTMLVideoElement ? media : null;

  const sorted = useMemo(() => [...partitions].sort((a, b) => a.start - b.start), [partitions]);
  const groupSizes = useMemo(() => {
    const sizes = new Map<string, number>();
    for (const p of partitions) sizes.set(p.group_id, (sizes.get(p.group_id) ?? 0) + 1);
    return sizes;
  }, [partitions]);

  const keepCount = sorted.filter((p) => partitionKind(p) === 'keep').length;
  const dropCount = sorted.filter((p) => partitionKind(p) === 'drop').length;
  const keepDuration = sorted.reduce((acc, p) => acc + (partitionKind(p) === 'keep' ? p.end - p.start : 0), 0);

  const [activeId, setActiveId] = useState<string | null>(null);
  useEffect(() => {
    if (!videoEl) return;
    function update() {
      if (!videoEl) return;
      const t = videoEl.currentTime;
      const hit = sorted.find((p) => t >= p.start - 0.01 && t < p.end - 0.01);
      setActiveId(hit ? hit.id : null);
    }
    update();
    videoEl.addEventListener('timeupdate', update);
    videoEl.addEventListener('seeked', update);
    return () => {
      videoEl.removeEventListener('timeupdate', update);
      videoEl.removeEventListener('seeked', update);
    };
  }, [videoEl, sorted]);

  // Auto-scroll the lane to follow the playhead while playing. Any manual
  // scroll suspends follow until the next play or seek.
  useEffect(() => {
    const lane = laneRef.current;
    if (!videoEl || !lane || duration <= 0) return;
    let raf = 0;
    let follow = true;
    let suppressScroll = 0;

    function scrollToPlayhead() {
      if (!lane || !videoEl) return;
      const topPx = (videoEl.currentTime / duration) * laneLengthPx;
      const visible = lane.clientHeight;
      const scrollTop = lane.scrollTop;
      if (topPx < scrollTop + 40 || topPx > scrollTop + visible - 80) {
        suppressScroll = 2;
        lane.scrollTop = Math.max(0, topPx - visible / 3);
      }
    }
    function tick() {
      if (videoEl && !videoEl.paused && follow) scrollToPlayhead();
      raf = requestAnimationFrame(tick);
    }
    function onUserScroll() {
      if (suppressScroll > 0) { suppressScroll--; return; }
      follow = false;
    }
    function onPlay() { follow = true; }
    function onSeeked() { follow = true; scrollToPlayhead(); }

    lane.addEventListener('scroll', onUserScroll, { passive: true });
    videoEl.addEventListener('play', onPlay);
    videoEl.addEventListener('seeked', onSeeked);
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      lane.removeEventListener('scroll', onUserScroll);
      videoEl.removeEventListener('play', onPlay);
      videoEl.removeEventListener('seeked', onSeeked);
    };
  }, [videoEl, duration, laneLengthPx]);

  const seekTo = useCallback((t: number) => {
    if (!videoEl || duration <= 0) return;
    onManualSeekIntent();
    videoEl.currentTime = Math.max(0, Math.min(duration, t));
  }, [videoEl, duration, onManualSeekIntent]);

  const playPartition = useCallback((id: string) => {
    if (!videoEl) return;
    const p = partitions.find((x) => x.id === id);
    if (!p) return;
    onManualSeekIntent();
    videoEl.currentTime = p.start;
    videoEl.play().catch(() => { /* ignore autoplay rejection */ });
  }, [videoEl, partitions, onManualSeekIntent]);

  const togglePartitionKeep = useCallback((id: string) => {
    const target = partitions.find((p) => p.id === id);
    if (!target) return;
    onPartitionsChange(partitions.map((p) => (p.id === id ? { ...p, keep: !p.keep } : p)));
  }, [partitions, onPartitionsChange]);

  // Drag handlers — bound once, read via dragRef so no stale closures.
  const onDragStart = useCallback((id: string, edge: 'start' | 'end' | 'body') => (e: React.PointerEvent<HTMLElement>) => {
    if (pxPerSec <= 0) return;
    e.stopPropagation();
    e.currentTarget.setPointerCapture(e.pointerId);
    const idx = sorted.findIndex((p) => p.id === id);
    const p = sorted[idx];
    if (!p) return;
    const prev = idx > 0 ? sorted[idx - 1] : null;
    const next = idx < sorted.length - 1 ? sorted[idx + 1] : null;
    dragRef.current = {
      id,
      edge,
      startY: e.clientY,
      origStart: p.start,
      origEnd: p.end,
      minStart: prev ? prev.start : 0,
      maxEnd: next ? next.end : duration,
      prevId: prev?.id ?? null,
      nextId: next?.id ?? null,
      moved: edge !== 'body',
    };
  }, [sorted, pxPerSec, duration]);

  const onDragMove = useCallback((e: React.PointerEvent<HTMLElement>) => {
    const d = dragRef.current;
    if (!d) return;
    const dy = e.clientY - d.startY;
    if (!d.moved) {
      if (Math.abs(dy) < DRAG_THRESHOLD_PX) return;
      d.moved = true;
      setIsDragging(true);
    }
    const dt = dy / pxPerSec;
    let newStart = d.origStart;
    let newEnd = d.origEnd;
    if (d.edge === 'start') {
      newStart = clamp(d.origStart + dt, d.minStart, d.origEnd - MIN_PARTITION_SECONDS);
    } else if (d.edge === 'end') {
      newEnd = clamp(d.origEnd + dt, d.origStart + MIN_PARTITION_SECONDS, d.maxEnd);
    } else {
      const origDur = d.origEnd - d.origStart;
      const maxDelta = d.maxEnd - d.origEnd;
      const minDelta = d.minStart - d.origStart;
      const clamped = clamp(dt, minDelta, maxDelta);
      newStart = d.origStart + clamped;
      newEnd = newStart + origDur;
    }
    onPartitionsChange(
      partitions.map((p) => {
        if (p.id === d.id) return { ...p, start: newStart, end: newEnd };
        if (d.prevId && p.id === d.prevId) return { ...p, end: newStart };
        if (d.nextId && p.id === d.nextId) return { ...p, start: newEnd };
        return p;
      }),
    );
  }, [partitions, pxPerSec, onPartitionsChange]);

  const onDragEnd = useCallback((e: React.PointerEvent<HTMLElement>) => {
    if (!dragRef.current) return;
    try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* already released */ }
    dragRef.current = null;
    setIsDragging(false);
  }, []);

  return (
    <div className={`timeline-container${isDragging ? ' dragging' : ''}`}>
      <div className="timeline-header">
        <span className="timeline-title">Timeline</span>
        <span className="timeline-stats">
          <span className="keep-stat">
            {keepCount} keep{dropCount > 0 ? ` · ${dropCount} retake${dropCount === 1 ? '' : 's'}` : ''}
            {' '}· {sorted.length} partitions
          </span>
          <span className="dur-stat">{fmt(keepDuration)} kept / {fmt(duration)} total</span>
        </span>
        <div className="timeline-zoom">
          <button className="zoom-btn" onClick={() => setPxPerSec((v) => Math.max(PX_PER_SEC_MIN, Math.round(v / 1.5)))} aria-label="Zoom out">−</button>
          <input
            type="range"
            className="zoom-slider"
            min={PX_PER_SEC_MIN}
            max={PX_PER_SEC_MAX}
            value={pxPerSec}
            onChange={(e) => setPxPerSec(Number(e.target.value))}
            aria-label="Zoom level"
          />
          <button className="zoom-btn" onClick={() => setPxPerSec((v) => Math.min(PX_PER_SEC_MAX, Math.round(v * 1.5)))} aria-label="Zoom in">+</button>
        </div>
      </div>

      <div className="timeline-legend">
        <span className="timeline-legend__item"><span className="timeline-legend__swatch keep" /> Keep</span>
        <span className="timeline-legend__item"><span className="timeline-legend__swatch drop" /> Retake / bad take</span>
        <span className="timeline-legend__item"><span className="timeline-legend__swatch cut" /> Silence</span>
      </div>

      <div className="timeline-hint">
        Click strip to seek · drag green block edges to trim · Keep/Cut toggles each take · <kbd>Space</kbd> play/pause
      </div>

      <div className="timeline-lane" ref={laneRef}>
        <div className="lane-inner" style={{ height: laneLengthPx }}>
          <div
            ref={stripRef}
            className="strip-col"
            style={{ width: STRIP_WIDTH_PX, height: laneLengthPx }}
            onClick={(e) => {
              if (!stripRef.current) return;
              const rect = stripRef.current.getBoundingClientRect();
              const y = e.clientY - rect.top;
              seekTo((y / laneLengthPx) * duration);
            }}
          >
            {sorted.map((p) => {
              const kind = partitionKind(p);
              if (kind === 'gap') return null;
              const top = p.start * pxPerSec;
              const height = Math.max(1, (p.end - p.start) * pxPerSec);
              if (kind === 'drop') {
                return (
                  <div
                    key={p.id}
                    className={`strip-block strip-block--drop${activeId === p.id ? ' active' : ''}`}
                    style={{ top, height }}
                    title="Retake — cut from export"
                  />
                );
              }
              return (
                <div
                  key={p.id}
                  className={`strip-block strip-block--keep${activeId === p.id ? ' active' : ''}`}
                  style={{ top, height }}
                >
                  <div
                    className="strip-handle strip-handle--top"
                    onPointerDown={onDragStart(p.id, 'start')}
                    onPointerMove={onDragMove}
                    onPointerUp={onDragEnd}
                    onPointerCancel={onDragEnd}
                    aria-label="Drag to trim start"
                  />
                  <div
                    className="strip-body"
                    onPointerDown={onDragStart(p.id, 'body')}
                    onPointerMove={onDragMove}
                    onPointerUp={onDragEnd}
                    onPointerCancel={onDragEnd}
                  />
                  <div
                    className="strip-handle strip-handle--bottom"
                    onPointerDown={onDragStart(p.id, 'end')}
                    onPointerMove={onDragMove}
                    onPointerUp={onDragEnd}
                    onPointerCancel={onDragEnd}
                    aria-label="Drag to trim end"
                  />
                </div>
              );
            })}
          </div>

          <div className="rows-col">
            {sorted.map((p) => {
              const top = p.start * pxPerSec;
              const height = Math.max(1, (p.end - p.start) * pxPerSec);
              const kind = partitionKind(p);
              const size = groupSizes.get(p.group_id) ?? 1;
              const label =
                kind === 'keep' && size > 1 ? `Last of ${size} takes`
                : kind === 'drop' ? `Take ${p.take_index + 1} of ${size}`
                : '';
              return (
                <SegRow
                  key={p.id}
                  partition={p}
                  kind={kind}
                  label={label}
                  top={top}
                  height={height}
                  active={activeId === p.id}
                  onPlay={playPartition}
                  onToggleKeep={togglePartitionKeep}
                  onSeek={(t) => seekTo(t)}
                />
              );
            })}
          </div>

          {videoEl && duration > 0 && (
            <Playhead videoEl={videoEl} duration={duration} laneLengthPx={laneLengthPx} />
          )}
        </div>
      </div>
    </div>
  );
}

interface SegRowProps {
  partition: Partition;
  kind: 'keep' | 'drop' | 'gap';
  label: string;
  top: number;
  height: number;
  active: boolean;
  onPlay: (id: string) => void;
  onToggleKeep: (id: string) => void;
  onSeek: (t: number) => void;
}

function SegRow({ partition: p, kind, label, top, height, active, onPlay, onToggleKeep, onSeek }: SegRowProps) {
  const rowClass = [
    'seg-row',
    `seg-row--${kind}`,
    active ? 'playing' : '',
  ].filter(Boolean).join(' ');

  if (kind === 'gap') {
    return (
      <div
        className={rowClass}
        style={{ top, height }}
        onClick={() => onSeek(p.start)}
        title="Click to seek here"
      >
        <button
          type="button"
          className={`seg-row__keep-toggle${p.keep ? ' seg-row__keep-toggle--keep' : ' seg-row__keep-toggle--cut'}`}
          onClick={(e) => { e.stopPropagation(); onToggleKeep(p.id); }}
          aria-pressed={p.keep}
          title={p.keep ? 'Keep (click to cut)' : 'Cut (click to keep)'}
        >
          {p.keep ? 'Keep' : 'Cut'}
        </button>
        <span className="seg-row__gap-label">(silence {fmt(p.end - p.start)})</span>
      </div>
    );
  }

  return (
    <div
      className={rowClass}
      style={{ top, height }}
      onClick={() => onPlay(p.id)}
    >
      <button
        className="seg-row__play"
        onClick={(e) => { e.stopPropagation(); onPlay(p.id); }}
        aria-label="Play this take"
      >
        <Play size={12} />
      </button>
      <button
        type="button"
        className={`seg-row__keep-toggle${p.keep ? ' seg-row__keep-toggle--keep' : ' seg-row__keep-toggle--cut'}`}
        onClick={(e) => { e.stopPropagation(); onToggleKeep(p.id); }}
        aria-pressed={p.keep}
        title={p.keep ? 'Keep (click to cut)' : 'Cut (click to keep)'}
      >
        {p.keep ? 'Keep' : 'Cut'}
      </button>
      <div className="seg-row__body">
        <div className="seg-row__text">
          {p.text || <em className="seg-row__text-empty">(no transcript)</em>}
        </div>
        <div className="seg-row__meta">
          <span>{fmt(p.start)}–{fmt(p.end)}</span>
          <span>{fmt(p.end - p.start)}</span>
          {label && <span className="seg-row__label">{label}</span>}
        </div>
      </div>
    </div>
  );
}

function Playhead({ videoEl, duration, laneLengthPx }: { videoEl: HTMLVideoElement; duration: number; laneLengthPx: number }) {
  const [topPx, setTopPx] = useState(0);
  useEffect(() => {
    if (!videoEl || duration <= 0 || laneLengthPx <= 0) return;
    let raf = 0;
    function tick() {
      setTopPx((videoEl.currentTime / duration) * laneLengthPx);
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [videoEl, duration, laneLengthPx]);
  return <div className="playhead" style={{ top: topPx }} />;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function fmt(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 10);
  return `${m}:${String(s).padStart(2, '0')}.${ms}`;
}
