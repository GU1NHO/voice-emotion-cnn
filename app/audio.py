from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 16_000
N_MELS = 64
MAX_SECONDS = 4.0

def to_mel(path: Path) -> np.ndarray:
    """Return a fixed-size log-mel spectrogram with one channel."""
    audio_path = _as_wav(path)
    signal, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=MAX_SECONDS)
    expected = int(SAMPLE_RATE * MAX_SECONDS)
    signal = np.pad(signal, (0, max(0, expected - len(signal))))[:expected]
    mel = librosa.feature.melspectrogram(y=signal, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Zero is the mean value, making zero-filled SpecAugment masks neutral.
    normalized = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return normalized.astype(np.float32)[None, ...]

def _as_wav(path: Path) -> Path:
    if path.suffix.lower() in {".wav", ".flac", ".ogg"}:
        return path
    converted = Path(tempfile.mkstemp(suffix=".wav")[1])
    result = subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ar", str(SAMPLE_RATE), "-ac", "1", str(converted)], capture_output=True, text=True)
    if result.returncode:
        converted.unlink(missing_ok=True)
        raise RuntimeError("Não foi possível converter o áudio. Instale o ffmpeg para enviar WebM/MP3.")
    return converted
