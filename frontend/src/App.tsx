import { useEffect, useRef, useState } from 'react';
import type { AppStage, SyncResult, AnalysisResult, Partition, SessionInfo } from './types/project';
import { syncFiles, analyzeAudio, exportVideo, listSessions } from './services/apiClient';
import FileUpload from './components/FileUpload/FileUpload';
import Timeline from './components/Timeline/Timeline';
import VideoPreview from './components/VideoPreview/VideoPreview';
import PlaybackControls from './components/VideoPreview/PlaybackControls';
import ExportPanel from './components/ExportPanel/ExportPanel';
import { useKeepsOnlyPlayback } from './hooks/useKeepsOnlyPlayback';
import './index.css';

const STORAGE_KEY = 'av-syncer:session';

interface PersistedState {
  stage: AppStage;
  syncResult: SyncResult | null;
  analysis: AnalysisResult | null;
  partitions: Partition[];
  downloadUrl: string | null;
  keepsOnly?: boolean;
}

function loadPersisted(): PersistedState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PersistedState;
    // Only restore if we actually reached a resumable stage
    if (!parsed.syncResult || !parsed.analysis) return null;
    // Pre-partition persisted state is incompatible with the new schema —
    // drop it rather than crash when AnalysisResult.partitions is missing.
    if (!Array.isArray(parsed.analysis.partitions) || !Array.isArray(parsed.partitions)) return null;
    // In-flight stages can't be resumed — drop back to editing if we got that far
    if (parsed.stage === 'syncing' || parsed.stage === 'analyzing' || parsed.stage === 'exporting') {
      parsed.stage = 'editing';
    }
    return parsed;
  } catch {
    return null;
  }
}

function keepPartitions(partitions: Partition[]): Partition[] {
  return partitions.filter((p) => p.keep && p.text.trim().length > 0);
}

export default function App() {
  const persisted = loadPersisted();
  const [stage, setStage] = useState<AppStage>(persisted?.stage ?? 'idle');
  const [error, setError] = useState<string | null>(null);
  const [syncResult, setSyncResult] = useState<SyncResult | null>(persisted?.syncResult ?? null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(persisted?.analysis ?? null);
  const [partitions, setPartitions] = useState<Partition[]>(persisted?.partitions ?? []);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(persisted?.downloadUrl ?? null);
  const [, setCurrentTime] = useState(0);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [keepsOnly, setKeepsOnly] = useState<boolean>(persisted?.keepsOnly ?? false);
  const [exportProgress, setExportProgress] = useState<number>(0);
  const manualSeekIntentRef = useRef(false);

  useKeepsOnlyPlayback(videoEl, partitions, keepsOnly, manualSeekIntentRef);

  function markManualSeekIntent() {
    manualSeekIntentRef.current = true;
  }

  useEffect(() => {
    if (!syncResult) {
      localStorage.removeItem(STORAGE_KEY);
      return;
    }
    const snapshot: PersistedState = { stage, syncResult, analysis, partitions, downloadUrl, keepsOnly };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  }, [stage, syncResult, analysis, partitions, downloadUrl, keepsOnly]);

  useEffect(() => {
    if (stage !== 'editing' || !videoEl) return;
    function handler(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return;
      if (!videoEl) return;
      if (e.code === 'Space') {
        e.preventDefault();
        if (videoEl.paused) videoEl.play(); else videoEl.pause();
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        // Arrow-key jump hops between keep chunks only; retakes/drops/silence
        // are navigable via the timeline but shouldn't steal the arrow keys.
        const keeps = keepPartitions(partitions).sort((a, b) => a.start - b.start);
        if (!keeps.length) return;
        e.preventDefault();
        const t = videoEl.currentTime;
        let next: Partition;
        if (e.key === 'ArrowDown') {
          next = keeps.find(p => p.start > t + 0.05) ?? keeps[0];
        } else {
          const earlier = keeps.filter(p => p.start < t - 0.05);
          next = earlier.length ? earlier[earlier.length - 1] : keeps[keeps.length - 1];
        }
        videoEl.currentTime = next.start;
        videoEl.play();
      }
    }
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [stage, videoEl, partitions]);

  async function handleFilesSelected(video: File, audio: File) {
    setError(null);
    try {
      setStage('syncing');
      const sync = await syncFiles(video, audio);
      setSyncResult(sync);

      setStage('analyzing');
      const result = await analyzeAudio(sync.session_id);
      setAnalysis(result);
      setPartitions(result.partitions);

      setStage('editing');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStage('idle');
    }
  }

  async function handleExport() {
    if (!syncResult) return;
    if (!keepPartitions(partitions).length) return;
    setError(null);
    try {
      setStage('exporting');
      setExportProgress(0);
      const result = await exportVideo(syncResult.session_id, partitions, setExportProgress);
      setDownloadUrl(result.download_url);
      setStage('done');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStage('editing');
    }
  }

  function handleReset() {
    setStage('idle');
    setSyncResult(null);
    setAnalysis(null);
    setPartitions([]);
    setDownloadUrl(null);
    setError(null);
    setCurrentTime(0);
    localStorage.removeItem(STORAGE_KEY);
  }

  async function handleReanalyze() {
    if (!syncResult) return;
    if (!confirm('Re-analyze will overwrite your manual keep/cut edits. Continue?')) return;
    setError(null);
    try {
      setStage('analyzing');
      const result = await analyzeAudio(syncResult.session_id);
      setAnalysis(result);
      setPartitions(result.partitions);
      setDownloadUrl(null);
      setStage('editing');
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStage('editing');
    }
  }

  function handleResume(info: SessionInfo) {
    if (!info.analysis) return;
    setError(null);
    setSyncResult(info.sync);
    setAnalysis(info.analysis);
    setPartitions(info.analysis.partitions);
    setDownloadUrl(null);
    setCurrentTime(0);
    setStage('editing');
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>AV Syncer</h1>
        <StageIndicator stage={stage} />
        {stage !== 'idle' && stage !== 'syncing' && stage !== 'analyzing' && (
          <>
            <button className="btn-ghost" onClick={handleReanalyze}>Re-analyze</button>
            <button className="btn-ghost" onClick={handleReset}>Start Over</button>
          </>
        )}
      </header>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <main className="app-main">
        {(stage === 'idle') && (
          <>
            <FileUpload onFilesSelected={handleFilesSelected} />
            <SessionPicker onResume={handleResume} />
          </>
        )}

        {(stage === 'syncing' || stage === 'analyzing') && (
          <div className="loading-panel">
            <div className="spinner" />
            <p>{stage === 'syncing' ? 'Syncing audio to video...' : 'Detecting bad takes (Whisper transcribing)...'}</p>
            {stage === 'analyzing' && (
              <p className="hint">This may take a minute — the Whisper medium model is transcribing your audio.</p>
            )}
          </div>
        )}

        {(stage === 'editing' || stage === 'exporting' || stage === 'done') && syncResult && analysis && (
          <div className="editor-layout">
            <div className="editor-left">
              <VideoPreview
                src={syncResult.synced_video_url}
                onVideoRef={setVideoEl}
                onTimeUpdate={setCurrentTime}
              />
              <PlaybackControls
                enabled={keepsOnly}
                onChange={setKeepsOnly}
                keepCount={keepPartitions(partitions).length}
              />
              <ExportPanel
                stage={stage}
                downloadUrl={downloadUrl}
                onExport={handleExport}
                keepCount={keepPartitions(partitions).length}
                exportProgress={exportProgress}
              />
            </div>
            <div className="editor-right">
              <Timeline
                audioUrl={syncResult.synced_video_url}
                duration={analysis.total_duration}
                partitions={partitions}
                media={videoEl}
                onPartitionsChange={setPartitions}
                onManualSeekIntent={markManualSeekIntent}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

const STAGE_LABELS: Record<AppStage, string> = {
  idle: 'Upload',
  uploading: 'Uploading',
  syncing: 'Syncing',
  analyzing: 'Analyzing',
  editing: 'Edit',
  exporting: 'Exporting',
  done: 'Done',
};
const VISIBLE_STAGES: AppStage[] = ['idle', 'syncing', 'analyzing', 'editing', 'exporting', 'done'];

function SessionPicker({ onResume }: { onResume: (s: SessionInfo) => void }) {
  const [items, setItems] = useState<SessionInfo[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listSessions()
      .then(data => { if (!cancelled) setItems(data); })
      .catch(e => { if (!cancelled) setErr(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, []);

  if (err) return <div className="session-picker error">Could not load sessions: {err}</div>;
  if (!items) return null;
  const resumable = items.filter(i => i.analysis);
  if (!resumable.length) return null;

  return (
    <div className="session-picker">
      <h3>Resume a previous session</h3>
      <ul className="session-list">
        {resumable.map(s => (
          <li key={s.session_id} className="session-row">
            <div className="session-meta">
              <div className="session-when">{formatDate(s.created_at)}</div>
              <div className="session-sub">
                {s.sync.duration.toFixed(1)}s · {sessionPartitionLabel(s.analysis?.partitions)} · <code>{s.session_id.slice(0, 8)}</code>
              </div>
            </div>
            <button className="btn-secondary" onClick={() => onResume(s)}>Resume</button>
          </li>
        ))}
      </ul>
    </div>
  );
}

function sessionPartitionLabel(partitions: Partition[] | undefined): string {
  if (!partitions) return '0 partitions';
  const total = partitions.length;
  const keep = partitions.filter((p) => p.keep && p.text.trim().length > 0).length;
  return `${keep} keep / ${total} partitions`;
}

function formatDate(iso: string): string {
  if (!iso) return 'unknown';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

function StageIndicator({ stage }: { stage: AppStage }) {
  const currentIdx = VISIBLE_STAGES.indexOf(stage);
  return (
    <div className="stage-indicator">
      {VISIBLE_STAGES.map((s, i) => (
        <div key={s} className={`stage-step ${i <= currentIdx ? 'active' : ''} ${s === stage ? 'current' : ''}`}>
          <div className="stage-dot">{i + 1}</div>
          <span>{STAGE_LABELS[s]}</span>
          {i < VISIBLE_STAGES.length - 1 && <div className="stage-line" />}
        </div>
      ))}
    </div>
  );
}
