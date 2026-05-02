import type { SyncResult, AnalysisResult, Partition, ExportResult, SessionInfo } from '../types/project';

const BASE = '/api';

async function checkOk(res: Response): Promise<Response> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? res.statusText);
  }
  return res;
}

export async function syncFiles(video: File, audio: File): Promise<SyncResult> {
  const form = new FormData();
  form.append('video', video);
  form.append('audio', audio);
  const res = await fetch(`${BASE}/sync`, { method: 'POST', body: form });
  await checkOk(res);
  return res.json();
}

export async function analyzeAudio(sessionId: string): Promise<AnalysisResult> {
  const form = new FormData();
  form.append('session_id', sessionId);
  const res = await fetch(`${BASE}/analyze`, { method: 'POST', body: form });
  await checkOk(res);
  return res.json();
}

export async function listSessions(): Promise<SessionInfo[]> {
  const res = await fetch(`${BASE}/sessions`);
  await checkOk(res);
  return res.json();
}

export async function exportVideo(
  sessionId: string,
  partitions: Partition[],
  onProgress: (pct: number) => void,
): Promise<ExportResult> {
  const res = await fetch(`${BASE}/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, partitions }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? res.statusText);
  }

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop()!;
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const payload = JSON.parse(line.slice(6)) as {
        progress?: number;
        done?: boolean;
        download_url?: string;
        error?: string;
      };
      if (payload.progress !== undefined) onProgress(payload.progress);
      if (payload.error) throw new Error(payload.error);
      if (payload.done && payload.download_url) return { download_url: payload.download_url };
    }
  }
  throw new Error('Export stream ended unexpectedly');
}
