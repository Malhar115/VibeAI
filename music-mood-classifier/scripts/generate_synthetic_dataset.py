import argparse
import os
import wave
import numpy as np


def write_wav(path: str, y: np.ndarray, sr: int):
    y = np.clip(y, -1.0, 1.0)
    data = (y * 32767.0).astype(np.int16).tobytes()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data)


def synth_calm(duration: float, sr: int, f: float = 220.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    env = 0.5 * (1.0 + 0.1 * np.sin(2 * np.pi * 0.2 * t))  # slow gentle AM
    y = env * np.sin(2 * np.pi * f * t)
    y += 0.01 * np.random.randn(len(t))
    y /= (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def synth_energetic(duration: float, sr: int, base_f: float = 440.0, bpm: float = 140.0) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * base_f * t) + 0.3 * np.sin(2 * np.pi * 3 * base_f * t)
    beat_hz = bpm / 60.0
    clicks = (np.sin(2 * np.pi * beat_hz * t) > 0.999).astype(np.float32)
    clicks = np.convolve(clicks, np.hanning(64), mode="same")
    y += 0.8 * clicks
    y += 0.02 * np.random.randn(len(t))
    y /= (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Generate a tiny synthetic dataset for mood classification.")
    ap.add_argument("--outdir", default="data/synth", help="Output root directory.")
    ap.add_argument("--n-per-class", type=int, default=8, help="Files per class.")
    ap.add_argument("--duration", type=float, default=5.0, help="Duration of each clip (seconds).")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate.")
    args = ap.parse_args()

    rng = np.random.RandomState(123)

    calm_dir = os.path.join(args.outdir, "calm")
    ener_dir = os.path.join(args.outdir, "energetic")

    os.makedirs(calm_dir, exist_ok=True)
    os.makedirs(ener_dir, exist_ok=True)

    for i in range(args.n_per_class):
        f = 200.0 + rng.uniform(-10, 10)
        y = synth_calm(args.duration, args.sr, f=f)
        write_wav(os.path.join(calm_dir, f"calm_{i+1:02d}.wav"), y, args.sr)

        base_f = 440.0 + rng.uniform(-20, 20)
        bpm = 135.0 + rng.uniform(-10, 10)
        y2 = synth_energetic(args.duration, args.sr, base_f=base_f, bpm=bpm)
        write_wav(os.path.join(ener_dir, f"energetic_{i+1:02d}.wav"), y2, args.sr)

    print(f"[v0] Generated dataset at {args.outdir}")
    print(f"[v0] Classes: calm, energetic | Files per class: {args.n_per_class} | Duration: {args.duration}s | SR: {args.sr}")


if __name__ == "__main__":
    main()
