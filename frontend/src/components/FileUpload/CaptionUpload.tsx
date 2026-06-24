import { Film, Upload } from 'lucide-react';
import { useRef, useState } from 'react';

interface Props {
  onVideoSelected: (video: File) => void;
}

export default function CaptionUpload({ onVideoSelected }: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) setVideo(file);
  }

  return (
    <div className="upload-panel">
      <h2>Auto Captions</h2>
      <div className="upload-row upload-row-single">
        <div
          className={`drop-zone ${video ? 'has-file' : ''}`}
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => { const f = e.target.files?.[0]; if (f) setVideo(f); }}
          />
          <Film size={32} />
          <div className="drop-label">Video File</div>
          {video ? (
            <div className="drop-filename">{video.name}</div>
          ) : (
            <div className="drop-hint">Click or drag &amp; drop</div>
          )}
        </div>
      </div>
      <button className="btn-primary" disabled={!video} onClick={() => video && onVideoSelected(video)}>
        <Upload size={16} />
        Upload &amp; Caption
      </button>
    </div>
  );
}
