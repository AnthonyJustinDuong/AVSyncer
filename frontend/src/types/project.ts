export interface Partition {
  id: string;
  start: number;
  end: number;
  text: string;
  group_id: string;
  take_index: number;
  keep: boolean;
}

export type PartitionKind = 'keep' | 'drop' | 'gap';

export function partitionKind(p: Partition): PartitionKind {
  if (!p.text.trim()) return 'gap';
  return p.keep ? 'keep' : 'drop';
}

export interface SyncResult {
  session_id: string;
  offset_seconds: number;
  synced_video_url: string;
  duration: number;
}

export interface AnalysisResult {
  session_id: string;
  partitions: Partition[];
  total_duration: number;
  audio_url: string;
}

export interface ExportResult {
  download_url: string;
}

export interface SessionInfo {
  session_id: string;
  created_at: string;
  sync: SyncResult;
  analysis: AnalysisResult | null;
}

export type AppStage =
  | 'idle'
  | 'uploading'
  | 'syncing'
  | 'analyzing'
  | 'editing'
  | 'exporting'
  | 'done';
