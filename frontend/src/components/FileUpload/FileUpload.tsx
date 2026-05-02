import { useRef, useState } from 'react';
import { Film, Mic, Upload } from 'lucide-react';

interface Props {
  onFilesSelected: (video: File, audio: File) => void;
}

export default function FileUpload({ onFilesSelected }: Props) {
  const [video, setVideo] = useState<File | null>(null);
  const [audio, setAudio] = useState<File | null>(null);
  const videoRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLInputElement>(null);

  function handleDrop(type: 'video' | 'audio', e: React.DragEvent) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    if (type === 'video') setVideo(file);
    else setAudio(file);
  }

  function handleSubmit() {
    if (video && audio) onFilesSelected(video, audio);
  }

  return (
    <div className="upload-panel">
      <h2>Upload Files</h2>
      <div className="upload-row">
        <DropZone
          label="Video File"
          icon={<Film size={32} />}
          file={video}
          accept="video/*"
          inputRef={videoRef}
          onDrop={(e) => handleDrop('video', e)}
          onChange={(f) => setVideo(f)}
        />
        <DropZone
          label="Audio File"
          icon={<Mic size={32} />}
          file={audio}
          accept="audio/*,video/*"
          inputRef={audioRef}
          onDrop={(e) => handleDrop('audio', e)}
          onChange={(f) => setAudio(f)}
        />
      </div>
      <button
        className="btn-primary"
        disabled={!video || !audio}
        onClick={handleSubmit}
      >
        <Upload size={16} />
        Sync &amp; Analyze
      </button>
    </div>
  );
}

interface DropZoneProps {
  label: string;
  icon: React.ReactNode;
  file: File | null;
  accept: string;
  inputRef: React.RefObject<HTMLInputElement>;
  onDrop: (e: React.DragEvent) => void;
  onChange: (f: File) => void;
}

function DropZone({ label, icon, file, accept, inputRef, onDrop, onChange }: DropZoneProps) {
  return (
    <div
      className={`drop-zone ${file ? 'has-file' : ''}`}
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: 'none' }}
        onChange={(e) => { const f = e.target.files?.[0]; if (f) onChange(f); }}
      />
      {icon}
      <div className="drop-label">{label}</div>
      {file ? (
        <div className="drop-filename">{file.name}</div>
      ) : (
        <div className="drop-hint">Click or drag &amp; drop</div>
      )}
    </div>
  );
}
