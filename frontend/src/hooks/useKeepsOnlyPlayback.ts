import { useEffect, useRef } from 'react';
import type { MutableRefObject } from 'react';
import type { Partition } from '../types/project';

/**
 * When `enabled`, forces <video> playback to only play inside keep partitions:
 * - If currentTime falls in a non-keep partition (retake or bad-take fragment),
 *   seek to the next keep partition's start.
 * - If past the last keep partition's end, pause.
 * Gap partitions (silence with keep=true) are treated as keeps — they are
 * part of the export and playing through them preserves natural pacing.
 */
export function useKeepsOnlyPlayback(
  videoEl: HTMLVideoElement | null,
  partitions: Partition[],
  enabled: boolean,
  manualSeekIntentRef?: MutableRefObject<boolean>,
) {
  // Latest partitions kept in a ref so the event handler doesn't need to
  // re-bind every time the user drags a keep chunk.
  const partitionsRef = useRef<Partition[]>(partitions);
  useEffect(() => {
    partitionsRef.current = partitions;
  }, [partitions]);

  useEffect(() => {
    if (!videoEl || !enabled) return;

    const EPS = 0.01;

    function sortedKeeps(): Partition[] {
      return partitionsRef.current
        .filter((p) => p.keep)
        .sort((a, b) => a.start - b.start);
    }

    function inKeep(t: number, keeps: Partition[]): boolean {
      return keeps.some((p) => t >= p.start - EPS && t < p.end - EPS);
    }

    function nextKeepStart(t: number, keeps: Partition[]): number | null {
      for (const p of keeps) {
        if (p.start > t - EPS) return p.start;
      }
      return null;
    }

    function correct() {
      if (!videoEl) return;
      if (videoEl.paused) return;
      if (manualSeekIntentRef?.current) {
        manualSeekIntentRef.current = false;
        return;
      }
      const keeps = sortedKeeps();
      if (!keeps.length) return;
      const t = videoEl.currentTime;
      if (inKeep(t, keeps)) return;
      const nxt = nextKeepStart(t, keeps);
      if (nxt == null) {
        if (!videoEl.paused) videoEl.pause();
        const last = keeps[keeps.length - 1];
        if (Math.abs(videoEl.currentTime - last.end) > EPS) {
          videoEl.currentTime = last.end;
        }
        return;
      }
      if (Math.abs(videoEl.currentTime - nxt) > EPS) {
        videoEl.currentTime = nxt;
      }
    }

    correct();

    videoEl.addEventListener('timeupdate', correct);
    videoEl.addEventListener('play', correct);

    return () => {
      videoEl.removeEventListener('timeupdate', correct);
      videoEl.removeEventListener('play', correct);
    };
  }, [videoEl, enabled, manualSeekIntentRef]);
}
