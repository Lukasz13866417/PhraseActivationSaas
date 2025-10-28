#!/usr/bin/env python3
import os, math, random, pathlib
import numpy as np
import soundfile as sf  # pip install soundfile (needs libsndfile)
from typing import Tuple

# ---------------- Configuration ----------------
SR = 16000
OUT_DIR = pathlib.Path("./dataset2/negative")
COUNT = 800              # change to 1000 if you like
DUR_RANGE = (0.8, 2.5)   # seconds, uniform
TAG = "artif_noise"      # naming convention for easy cleanup
RNG_SEED = None          # set int for reproducibility, else None
# ------------------------------------------------

def ensure_dir(p: pathlib.Path):
    p.mkdir(parents=True, exist_ok=True)

def rand_dur(rng: np.random.Generator) -> int:
    d = rng.uniform(*DUR_RANGE)
    return int(round(d * SR))

def normalize(x: np.ndarray, peak=0.98) -> np.ndarray:
    m = np.max(np.abs(x)) + 1e-12
    return np.clip(x * (peak / m), -1.0, 1.0)

# ---- Noise/tonal generators (nonsensical, non-speech) ----
def white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32)

def brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.standard_normal(n).astype(np.float32)
    y = np.cumsum(x)     # integrate → Brownian
    return y

def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    # Voss-McCartney: average several random LFs
    n_sources = 16
    out = np.zeros(n, dtype=np.float32)
    vals = rng.standard_normal(n_sources).astype(np.float32)
    counters = np.zeros(n_sources, dtype=np.int64)
    for i in range(n):
        idx = rng.integers(0, n_sources)
        counters[idx] += 1
        vals[idx] = rng.standard_normal()
        out[i] = vals.mean()
    return out

def sine_beeps(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    t = np.arange(n) / SR
    k_beeps = rng.integers(2, 6)
    for _ in range(k_beeps):
        f = float(rng.uniform(200, 3500))
        dur = rng.uniform(0.05, 0.25)
        start = rng.integers(0, max(1, n - int(dur*SR)))
        end = min(n, start + int(dur * SR))
        env = np.ones(end - start, dtype=np.float32)
        a = max(0.01, rng.uniform(0.005, 0.02))
        r = max(0.02, rng.uniform(0.01, 0.05))
        # simple AR envelope
        L = end - start
        e = np.ones(L, dtype=np.float32)
        aL = min(int(a*SR), L)
        rL = min(int(r*SR), L)
        if aL > 0:
            e[:aL] = np.linspace(0, 1, aL, dtype=np.float32)
        if rL > 0:
            e[-rL:] = np.linspace(1, 0, rL, dtype=np.float32)
        phase = 2*np.pi*f*t[:L] + rng.uniform(0, 2*np.pi)
        x[start:end] += (0.7 * np.sin(phase) * e).astype(np.float32)
    return x

def fm_warble(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f0 = rng.uniform(150, 600)
    f1 = rng.uniform(800, 3000)
    fm = rng.uniform(2, 12)  # Hz
    beta = rng.uniform(10, 80)
    f_inst = f0 + (f1 - f0) * (t / t[-1])
    phase = 2*np.pi*np.cumsum(f_inst/SR) + beta*np.sin(2*np.pi*fm*t)
    return 0.8*np.sin(phase).astype(np.float32)

def clicks(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    k = rng.integers(5, 25)
    for _ in range(k):
        pos = rng.integers(0, n)
        amp = rng.uniform(0.2, 1.0)
        width = rng.integers(1, int(0.003*SR))  # up to ~3 ms
        start = max(0, pos - width//2)
        end = min(n, start + width)
        x[start:end] += amp * (rng.standard_normal(end-start)).astype(np.float32)
    # light smoothing
    kernel = np.hanning(9).astype(np.float32); kernel /= kernel.sum()
    return np.convolve(x, kernel, mode="same").astype(np.float32)

def chirp_sweep(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f0 = rng.uniform(100, 400)
    f1 = rng.uniform(3000, 6000)
    k = (f1/f0)**(1/max(1e-6, t[-1]))
    phase = 2*np.pi*f0*(k**t - 1)/np.log(k)
    return 0.8*np.sin(phase + rng.uniform(0, 2*np.pi)).astype(np.float32)

def shape_and_gain(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # random overall gain -14..-2 dB
    gain_db = rng.uniform(-14, -2)
    x = x * (10**(gain_db/20))
    # optional bandpass to avoid sounding speechy
    if rng.random() < 0.7:
        # simple FIR via convolution with short windowed-sinc bandpass
        from numpy.fft import rfft, irfft, rfftfreq
        X = rfft(x)
        freqs = rfftfreq(x.size, 1/SR)
        lo = rng.uniform(50, 300)
        hi = rng.uniform(2000, 7000)
        lo, hi = min(lo, hi*0.7), max(hi, lo*1.3)
        H = np.logical_and(freqs >= lo, freqs <= hi).astype(np.float32)
        X *= H
        x = irfft(X, n=x.size).astype(np.float32)
    return x

GENS = [
    white_noise, pink_noise, brown_noise,
    sine_beeps, fm_warble, clicks, chirp_sweep
]

def main():
    rng = np.random.default_rng(RNG_SEED)
    ensure_dir(OUT_DIR)
    created = 0
    for i in range(COUNT):
        n = rand_dur(rng)
        g = random.choice(GENS)
        x = g(n, rng)
        x = shape_and_gain(x, rng)
        x = normalize(x)
        fn = f"{TAG}_{i:05d}.wav"
        path = OUT_DIR / fn
        sf.write(path, x, SR, subtype="PCM_16")
        created += 1
    print(f"Generated {created} nonsensical negatives in {OUT_DIR} (prefix '{TAG}_').")

if __name__ == "__main__":
    main()
