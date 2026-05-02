import './PlaybackControls.css';

interface Props {
  enabled: boolean;
  onChange: (v: boolean) => void;
  keepCount: number;
}

export default function PlaybackControls({ enabled, onChange, keepCount }: Props) {
  const disabled = keepCount === 0;
  return (
    <div className={`playback-controls${disabled ? ' disabled' : ''}`}>
      <label className="keeps-only-toggle">
        <input
          type="checkbox"
          checked={enabled}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="keeps-only-track" aria-hidden>
          <span className="keeps-only-thumb" />
        </span>
        <span className="keeps-only-label">Preview keeps only</span>
      </label>
      <div className="playback-controls-hint">Skips cuts during playback</div>
    </div>
  );
}
