"""Audio-video sync: find time offset between two audio recordings via cross-correlation."""
import os
import tempfile
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
from services.ffmpeg_runner import extract_audio


SAMPLE_RATE = 16000
MAX_CORRELATION_DURATION = 60.0  # seconds — only use first N seconds for correlation


def _load_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as a mono float32 array."""
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Normalize integer PCM to float32 in [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)
    return data, sr


def _extract_to_wav(input_path: str, out_wav: str) -> None:
    """Extract audio from any media file as 16 kHz mono WAV."""
    extract_audio(input_path, out_wav)


def find_sync_offset(video_path: str, external_audio_path: str) -> float:
    """
    Find the time offset (in seconds) between the audio embedded in the video
    and the external audio file using FFT-based cross-correlation.

    Returns offset_seconds such that:
      positive -> external audio starts later than video audio (delay external)
      negative -> external audio starts earlier than video audio (trim external)
    """
    with tempfile.TemporaryDirectory() as tmp:
        video_wav = os.path.join(tmp, "video_audio.wav")
        ext_wav = os.path.join(tmp, "external_audio.wav")
        _extract_to_wav(video_path, video_wav)
        _extract_to_wav(external_audio_path, ext_wav)

        video_sig, sr = _load_wav_mono(video_wav)
        ext_sig, _ = _load_wav_mono(ext_wav)

        # Limit the signals to the first N seconds for speed — the offset should
        # be apparent early in the recording and long signals make FFT slow.
        max_samples = int(MAX_CORRELATION_DURATION * sr)
        video_sig = video_sig[:max_samples]
        ext_sig = ext_sig[:max_samples]

        # Normalize: subtract mean, divide by RMS. Makes correlation scale-invariant.
        video_sig = _normalize(video_sig)
        ext_sig = _normalize(ext_sig)

        # Cross-correlation via FFT. fftconvolve with reversed signal == correlate.
        # correlation[k] is high when ext_sig shifted by (k - (N-1)) matches video_sig.
        correlation = fftconvolve(video_sig, ext_sig[::-1], mode="full")

        peak_idx = int(np.argmax(np.abs(correlation)))
        lag_samples = peak_idx - (len(ext_sig) - 1)
        # Positive lag means ext_sig occurs later in video_sig → external audio is behind
        offset_seconds = lag_samples / sr

        return float(offset_seconds)


def _normalize(sig: np.ndarray) -> np.ndarray:
    sig = sig - np.mean(sig)
    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms > 1e-9:
        sig = sig / rms
    return sig
