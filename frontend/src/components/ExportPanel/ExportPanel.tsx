import { Download, Scissors } from 'lucide-react';
import type { AppStage } from '../../types/project';

interface Props {
  stage: AppStage;
  downloadUrl: string | null;
  onExport: () => void;
  keepCount: number;
  exportProgress: number;
}

export default function ExportPanel({ stage, downloadUrl, onExport, keepCount, exportProgress }: Props) {
  const isExporting = stage === 'exporting';
  const isDone = stage === 'done';

  return (
    <div className="export-panel">
      <div className="export-summary">
        <span className="keep-badge">{keepCount} keep partition{keepCount === 1 ? '' : 's'}</span>
      </div>

      {!isDone && (
        <button
          className="btn-primary btn-export"
          disabled={isExporting || keepCount === 0}
          onClick={onExport}
        >
          {isExporting ? (
            <><div className="spinner-sm" /> Exporting{exportProgress > 0 ? ` ${exportProgress}%` : '...'}</>
          ) : (
            <><Scissors size={16} /> Export Final Video</>
          )}
        </button>
      )}

      {isExporting && exportProgress > 0 && (
        <div className="export-progress-bar">
          <div className="export-progress-fill" style={{ width: `${exportProgress}%` }} />
        </div>
      )}

      {isDone && downloadUrl && (
        <a
          className="btn-primary btn-download"
          href={downloadUrl}
          download="export.mp4"
        >
          <Download size={16} />
          Download MP4
        </a>
      )}
    </div>
  );
}
