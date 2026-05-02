import { useRef } from 'react';

interface Props {
  src: string;
  onVideoRef: (el: HTMLVideoElement | null) => void;
  onTimeUpdate: (t: number) => void;
}

export default function VideoPreview({ src, onVideoRef, onTimeUpdate }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);

  function setRef(el: HTMLVideoElement | null) {
    videoRef.current = el;
    onVideoRef(el);
  }

  function handleTimeUpdate() {
    const video = videoRef.current;
    if (video) onTimeUpdate(video.currentTime);
  }

  return (
    <div className="video-preview">
      <video
        ref={setRef}
        src={src}
        controls
        onTimeUpdate={handleTimeUpdate}
        style={{ width: '100%', borderRadius: '8px', background: '#000' }}
      />
    </div>
  );
}
