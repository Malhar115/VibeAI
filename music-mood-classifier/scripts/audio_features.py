import os
import math
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np

# Optional: librosa for broader codec support (mp3, flac) and extra features.
HAS_LIBROSA = False
try:
    import librosa  # type: ignore
    HAS_LIBROSA = True
except Exception:
    HAS_LIBROSA = False

EPS = 1e-10


def _resample_linear(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return y
    duration = y.shape[0] / float(src_sr)
    dst_len = int(round(duration * dst_sr))
    if dst_len <= 1:
        return y
    x_old = np.linspace(0.0, duration, num=y.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, y).astype(np.float32)


def _read_wav_basic(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    # Minimal WAV PCM reader (16/32-bit), mono/stereo -> mono float32 [-1,1]
    import wave

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        frames = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
        scale = 32768.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    data = np.frombuffer(frames, dtype=dtype).astype(np.float32) / scale
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    y = data.astype(np.float32)
    y = _resample_linear(y, framerate, target_sr)
    if np.max(np.abs(y)) >= EPS:
        y = (y / (np.max(np.abs(y)) + EPS)).astype(np.float32)
    return y, target_sr


def load_audio(path: str, target_sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono float32 at target_sr.
    - Uses librosa if available (supports mp3/flac/ogg/m4a).
    - Falls back to basic WAV reader otherwise (.wav only).
    """
    ext = os.path.splitext(path)[1].lower()
    if HAS_LIBROSA:
        y, sr = librosa.load(path, sr=target_sr, mono=True)
        y = y.astype(np.float32)
        return y, sr
    else:
        if ext != ".wav":
            raise RuntimeError(
                f"librosa not available; only .wav supported. Got: {ext} ({path})"
            )
        return _read_wav_basic(path, target_sr=target_sr)


def frame_signal(y: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if len(y) < frame_size:
        y = np.pad(y, (0, frame_size - len(y)), mode="constant")
    n_frames = 1 + (len(y) - frame_size) // hop_size
    idx = np.tile(np.arange(0, frame_size), (n_frames, 1)) + np.tile(
        np.arange(0, n_frames * hop_size, hop_size), (frame_size, 1)
    ).T
    frames = y[idx]
    window = np.hanning(frame_size).astype(np.float32)
    return (frames * window).astype(np.float32)


def compute_zcr(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = 1
    changes = (np.diff(signs, axis=1) != 0).astype(np.float32)
    zcr = np.sum(changes, axis=1) / (2.0 * frames.shape[1])
    return zcr


def compute_energy(frames: np.ndarray) -> np.ndarray:
    return np.mean(frames**2, axis=1)


def _rfft_mag(frames: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec).astype(np.float32)
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    return mag, freqs


def compute_spectral_centroid(frames: np.ndarray, sr: int) -> np.ndarray:
    mag, freqs = _rfft_mag(frames, sr)
    num = (mag * freqs[None, :]).sum(axis=1)
    den = (mag.sum(axis=1) + EPS)
    return num / den


def compute_spectral_bandwidth(frames: np.ndarray, sr: int) -> np.ndarray:
    mag, freqs = _rfft_mag(frames, sr)
    centroid = compute_spectral_centroid(frames, sr)[:, None]
    var = ((mag * ((freqs[None, :] - centroid) ** 2)).sum(axis=1)) / (mag.sum(axis=1) + EPS)
    return np.sqrt(np.maximum(var, 0.0))


def compute_spectral_rolloff(frames: np.ndarray, sr: int, roll_percent: float = 0.85) -> np.ndarray:
    mag, freqs = _rfft_mag(frames, sr)
    cum = np.cumsum(mag, axis=1)
    total = (mag.sum(axis=1, keepdims=True) + EPS)
    thresh = roll_percent * total
    idx = (cum >= thresh).argmax(axis=1)
    return freqs[idx]


def compute_spectral_flatness(frames: np.ndarray, sr: int) -> np.ndarray:
    mag, _ = _rfft_mag(frames, sr)
    geo = np.exp(np.mean(np.log(mag + EPS), axis=1))
    arith = np.mean(mag + EPS, axis=1)
    return (geo / arith).astype(np.float32)


def extract_features(
    path: str,
    target_sr: int = 22050,
    frame_size: int = 1024,
    hop_size: int = 512,
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      - feature vector of shape (n_features,)
      - corresponding feature names
    Core: ZCR, Energy, Spectral Centroid/Bandwidth/Rolloff/Flatness (mean/std per feature)
    Optional (librosa): MFCC(13), Chroma(12), Spectral Contrast (mean/std per band)
    """
    y, sr = load_audio(path, target_sr=target_sr)
    frames = frame_signal(y, frame_size=frame_size, hop_size=hop_size)

    feats: Dict[str, Any] = {}

    # Core features (NumPy-only)
    zcr = compute_zcr(frames)
    energy = compute_energy(frames)
    sc = compute_spectral_centroid(frames, sr)
    sbw = compute_spectral_bandwidth(frames, sr)
    sro = compute_spectral_rolloff(frames, sr, 0.85)
    sflat = compute_spectral_flatness(frames, sr)

    def agg(name: str, arr: np.ndarray):
        feats[f"{name}_mean"] = float(np.mean(arr))
        feats[f"{name}_std"] = float(np.std(arr) + EPS)

    agg("zcr", zcr)
    agg("energy", energy)
    agg("spec_centroid", sc)
    agg("spec_bandwidth", sbw)
    agg("spec_rolloff85", sro)
    agg("spec_flatness", sflat)

    # librosa extras (optional)
    if HAS_LIBROSA:
        try:
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_size, n_fft=frame_size)
            for i in range(mfcc.shape[0]):
                agg(f"mfcc_{i+1}", mfcc[i, :])

            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_size, n_fft=frame_size)
            for i in range(chroma.shape[0]):
                agg(f"chroma_{i+1}", chroma[i, :])

            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_size, n_fft=frame_size)
            for i in range(contrast.shape[0]):
                agg(f"spec_contrast_{i+1}", contrast[i, :])
        except Exception as e:
            warnings.warn(f"librosa feature extraction failed for {path}: {e}")

    names = list(feats.keys())
    vec = np.array([feats[n] for n in names], dtype=np.float32)
    return vec, names
