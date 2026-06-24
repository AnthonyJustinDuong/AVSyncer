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

export interface CaptionWord {
  id: string;
  start: number;
  end: number;
  text: string;
}

export interface CaptionCue {
  id: string;
  start: number;
  end: number;
  words: CaptionWord[];
}

export interface CaptionStyle {
  x: number;
  y: number;
  max_width: number;
  font_size: number;
  base_color: string;
  highlight_color: string;
  outline_color: string;
  outline_width: number;
  shadow_color: string;
  shadow_opacity: number;
  shadow_blur: number;
  shadow_offset: number;
  align: 'left' | 'center' | 'right';
  highlight_mode: 'progressive' | 'active_word' | 'pop_word';
}

export interface CaptionProject {
  session_id: string;
  created_at: string;
  video_url: string;
  duration: number;
  cues: CaptionCue[];
  style: CaptionStyle;
}

export type AppStage =
  | 'idle'
  | 'uploading'
  | 'syncing'
  | 'analyzing'
  | 'editing'
  | 'exporting'
  | 'done';
