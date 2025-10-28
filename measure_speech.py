#!/usr/bin/env python3
import argparse, glob, math, os
import numpy as np

try:
    import soundfile as sf  # pip install soundfile
    read_wav = lambda p: sf.read(p, dtype="float32")
except Exception:
    # Fallback to scipy if soundfile isn't available
    from scipy.io import wavfile
    def read_wav(p):
        sr, x = wavfile.read(p)
        # convert to float32 [-1, 1]
        if x.dtype.kind in "iu":
            maxv = np.iinfo(x.dtype).max
            x = x.astype(np.float32) / maxv
        else:
            x = x.astype(np.float32)
        return x, sr

def frame_rms(x, frame_len, hop):
    # Pad so we include the tail; then strided framing
    pad = (frame_len - (len(x) - frame_len) % hop - 1) % hop
    x = np.pad(x, (0, pad), mode="constant")
    n_frames = 1 + (len(x) - frame_len) // hop
    shape = (n_frames, frame_len)
    strides = (x.strides[0]*hop, x.strides[0])
    frames = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    return rms

def detect_voiced_indices(x, sr,
                          frame_ms=20,
                          hop_ms=10,
                          noise_probe_ms=400,   # how much from start & end to learn “silence”
                          thr_db_above_noise=6, # lower this to overshoot more
                          min_region_ms=40,     # drop tiny blips
                          close_gap_ms=60,      # fill small gaps inside speech
                          pad_ms=80):           # pad both sides to overshoot
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    # High-pass a touch (remove DC); 1st-order filter
    x = x - np.mean(x)

    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    frame_len = max(frame_len, 1)
    hop = max(hop, 1)

    rms = frame_rms(x, frame_len, hop)
    rms_db = 20*np.log10(rms + 1e-12)

    # Estimate “silence” from start and end (clips start/end with silence per your spec)
    probe_samps = int(sr * noise_probe_ms / 1000)
    start_probe = x[:probe_samps] if probe_samps > 0 else x[:1]
    end_probe   = x[-probe_samps:] if probe_samps > 0 else x[-1:]
    probe = np.concatenate([start_probe, end_probe]) if len(x) > probe_samps else x

    probe_rms = frame_rms(probe, frame_len, hop)
    noise_db = np.percentile(20*np.log10(probe_rms + 1e-12), 95)  # conservative
    thr_db = noise_db + thr_db_above_noise

    # Primary detection
    voiced = rms_db > thr_db

    # Morphological smoothing: close short gaps, drop short islands
    def ms_to_frames(ms): return max(1, int(round(ms / hop_ms)))
    close_n = ms_to_frames(close_gap_ms)
    min_region_n = ms_to_frames(min_region_ms)
    pad_n = ms_to_frames(pad_ms)

    # Close small gaps (binary closing = dilate then erode)
    # Implement: fill gaps shorter than close_n
    i = 0
    while i < len(voiced):
        if not voiced[i]:
            j = i
            while j < len(voiced) and not voiced[j]:
                j += 1
            gap_len = j - i
            if 0 < gap_len <= close_n:
                voiced[i:j] = True
            i = j
        else:
            i += 1

    # Remove tiny voiced islands
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            seg_len = j - i
            if seg_len < min_region_n:
                voiced[i:j] = False
            i = j
        else:
            i += 1

    # Pad both sides to overshoot
    if np.any(voiced):
        idx = np.flatnonzero(voiced)
        start = max(0, idx[0] - pad_n)
        end = min(len(voiced)-1, idx[-1] + pad_n)
        voiced[start:end+1] = True

    # Final duration in seconds
    if not np.any(voiced):
        return 0.0
    idx = np.flatnonzero(voiced)
    # convert frame index span to time; align to frame centers via hop
    duration_sec = (idx[-1] - idx[0] + 1) * (hop / sr)
    return duration_sec

def summarize_durations(durations):
    arr = np.array(durations, dtype=float)
    stats = {
        "count": int(arr.size),
        "p90_s": float(np.percentile(arr, 90)) if arr.size else 0.0,
        "p95_s": float(np.percentile(arr, 95)) if arr.size else 0.0,
        "max_s": float(np.max(arr)) if arr.size else 0.0,
        "mean_s": float(np.mean(arr)) if arr.size else 0.0,
        "std_s": float(np.std(arr)) if arr.size else 0.0,
    }
    return stats

def main():
    ap = argparse.ArgumentParser(description="Measure spoken durations in WAVs and report p90/p95/max.")
    ap.add_argument("inputs", nargs="+", help="WAV files or globs (e.g., data/*.wav)")
    ap.add_argument("--pad-ms", type=int, default=80, help="Padding around speech to overshoot (default 80).")
    ap.add_argument("--thr-db-above-noise", type=float, default=6.0, help="Threshold above noise floor in dB (default 6). Lower => more overshoot.")
    args = ap.parse_args()

    # Expand globs
    paths = []
    for pat in args.inputs:
        paths.extend(glob.glob(pat))
    paths = sorted(set(p for p in paths if os.path.isfile(p)))

    if not paths:
        print("No files found.")
        return

    durs = []
    for p in paths:
        x, sr = read_wav(p)
        dur = detect_voiced_indices(
            x, sr,
            thr_db_above_noise=args.thr_db_above_noise,
            pad_ms=args.pad_ms
        )
        durs.append(dur)
        print(f"{os.path.basename(p)}\t{dur*1000:.1f} ms")

    stats = summarize_durations(durs)
    print("\nSummary across files:")
    print(f"count = {stats['count']}")
    print(f"p90   = {stats['p90_s']*1000:.1f} ms")
    print(f"p95   = {stats['p95_s']*1000:.1f} ms")
    print(f"max   = {stats['max_s']*1000:.1f} ms")
    print(f"mean  = {stats['mean_s']*1000:.1f} ms ± {stats['std_s']*1000:.1f} ms")

if __name__ == "__main__":
    main()
