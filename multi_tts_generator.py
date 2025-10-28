from __future__ import annotations
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import torchaudio
import torch

# Piper/Coqui
from piper_api.piper_model_getter import VoiceEntry as PiperVoiceEntry
from piper_api.piper_model_getter import list_piper_voices, synthesize_to_wav as piper_synth

from coqui_api.coqui_model_getter import CoquiModelEntry
from coqui_api.coqui_model_getter import list_coqui_models, synthesize_to_wav as coqui_synth

# ElevenLabs
from eleven_api.eleven_model_getter import ElevenVoiceEntry
from eleven_api.eleven_model_getter import list_eleven_voices, synthesize_to_wav as eleven_synth


@dataclass
class NegativeConfig:
    # Long utterances to sample from (generic text, no keyword)
    long_texts: List[str]
    # Probability that a negative will be a random 0-3s utterance vs a random clip from a long utterance
    p_short_utterance: float = 0.5
    # Clip durations & ranges
    short_min_s: float = 0.8
    short_max_s: float = 3.0
    clip_min_s: float = 0.7
    clip_max_s: float = 2.5


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rand_time_range(total_s: float, min_len: float, max_len: float, rng: random.Random) -> Tuple[float, float]:
    length = rng.uniform(min_len, max_len)
    if length >= total_s:
        return 0.0, max(0.0, total_s)
    start = rng.uniform(0.0, max(0.0, total_s - length))
    return start, start + length


def _slice_wav(in_wav: Path, out_wav: Path, start_s: float, end_s: float) -> None:
    wav, sr = torchaudio.load(str(in_wav))
    start = int(start_s * sr)
    end = int(end_s * sr)
    start = max(0, start); end = max(start + 1, end)
    wav = wav[..., start:end]
    torchaudio.save(str(out_wav), wav, sr)


def _safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:80]


def _normalize_space(s: str) -> str:
    return " ".join(s.split())


def generate_all(
    *,
    phrase: str,
    lang: str,
    out_dir: str | Path,
    positives_per_engine: int = 10,
    negatives_total: int = 60,
    max_voices: int = 2,
    resample_hz: Optional[int] = 16000,
    seed: Optional[int] = None,
    negative_cfg: Optional[NegativeConfig] = None,
) -> dict:
    """
    Generate positives (exact phrase) and negatives (clips & short utterances) using Piper, Coqui, and Eleven.

    Layout:
      out_dir/
        positive/
          piper_*.wav, coqui_*.wav, eleven_*.wav
        negative/
          clip_*.wav, short_*.wav
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    pos_dir = out_dir / "positive"; _ensure_dir(pos_dir)
    neg_dir = out_dir / "negative"; _ensure_dir(neg_dir)

    phrase = _normalize_space(phrase)

    # Discover voices/models for each engine
    piper_vs = list_piper_voices(lang=lang)
    if max_voices is not None: piper_vs = piper_vs[:max_voices]

    try:
        coqui_ms = list_coqui_models(lang=lang)
        if max_voices is not None: coqui_ms = coqui_ms[:max_voices]
    except Exception:
        coqui_ms = []

    try:
        eleven_vs = list_eleven_voices()
        if max_voices is not None: eleven_vs = eleven_vs[:max_voices]
    except Exception:
        eleven_vs = []

    created_pos: List[Path] = []

    # Positives: phrase only
    for i in range(positives_per_engine):
        # Piper
        if piper_vs:
            v = piper_vs[i % len(piper_vs)]
            out = pos_dir / f"piper_{_safe_slug(v.pretty_id)}_{i:04d}.wav"
            piper_synth(phrase, v, out, resample_hz=resample_hz)
            created_pos.append(out)

        # Coqui
        if coqui_ms:
            m = coqui_ms[i % len(coqui_ms)]
            out = pos_dir / f"coqui_{_safe_slug(m.pretty_id)}_{i:04d}.wav"
            coqui_synth(phrase, m, out, resample_hz=resample_hz)
            created_pos.append(out)

        # Eleven
        if eleven_vs:
            e = eleven_vs[i % len(eleven_vs)]
            out = pos_dir / f"eleven_{_safe_slug(e.pretty_id)}_{i:04d}.wav"
            try:
                eleven_synth(phrase, e, out, resample_hz=resample_hz)
                created_pos.append(out)
            except Exception:
                pass

    # Negatives
    neg_cfg = negative_cfg or NegativeConfig(long_texts=[
        "Please follow the safety protocols and keep your distance from the machinery.",
        "In the event of an emergency, proceed to the nearest exit in an orderly fashion.",
        "Our service hours are from nine A.M. to six P.M. Monday through Friday.",
        "For assistance, contact the help desk or check the online documentation portal.",
        "The fitnessgram pacer test is a 30 second test that measures your aerobic fitness.",
        "We will do what we must, but we do it for Aiur, not you.",
        "Do you seek knowledge of time travel?",
        "This is not a drill, this is not a drill, this is not a drill.",
        "A wall of text is an excessively long post to a noticeboard or talk page discussion, which can often be so long that some don't read it.",
        "A text file that contains only text, with no formatting, is called plain text, and is the most basic way to store text.",
        "The quick brown fox jumps over the lazy dog.",
        "How to survive a wall of text? Read surrounding posts, or skim to determine whether the long post is largely substantive or mostly irrelevant. If it is the latter, apply trout and other remedies in suitable proportion. Simplest is just to ignore it if it's not relevant to you."
    ])

    def synth_any(text: str, idx: int, prefix: str) -> Optional[Path]:
        # rotate engines to create negatives too
        sel = idx % 3
        if sel == 0 and piper_vs:
            v = piper_vs[idx % len(piper_vs)]
            out = neg_dir / f"{prefix}_piper_{idx:05d}.wav"
            piper_synth(text, v, out, resample_hz=resample_hz); return out
        if sel == 1 and coqui_ms:
            m = coqui_ms[idx % len(coqui_ms)]
            out = neg_dir / f"{prefix}_coqui_{idx:05d}.wav"
            coqui_synth(text, m, out, resample_hz=resample_hz); return out
        if sel == 2 and eleven_vs:
            e = eleven_vs[idx % len(eleven_vs)]
            out = neg_dir / f"{prefix}_eleven_{idx:05d}.wav"
            try:
                eleven_synth(text, e, out, resample_hz=resample_hz); return out
            except Exception:
                return None
        # fallback to piper if available
        if piper_vs:
            v = piper_vs[idx % len(piper_vs)]
            out = neg_dir / f"{prefix}_piper_{idx:05d}.wav"
            piper_synth(text, v, out, resample_hz=resample_hz); return out
        return None

    created_neg: List[Path] = []

    for k in range(negatives_total):
        if random.random() < neg_cfg.p_short_utterance:
            # short 0-3s utterance, sampled by generating a short sentence (no keyword)
            text = random.choice(neg_cfg.long_texts)
            out_full = synth_any(text, k, prefix="short")
            if out_full is None: continue
            # slice a short random segment from generated audio
            wav, sr = torchaudio.load(str(out_full))
            total_s = wav.shape[-1] / sr
            a, b = _rand_time_range(total_s, neg_cfg.short_min_s, neg_cfg.short_max_s, rng)
            out_clip = neg_dir / f"shortclip_{k:05d}.wav"
            _slice_wav(out_full, out_clip, a, min(b, total_s))
            created_neg.append(out_clip)
        else:
            # random clip from a longer utterance
            text = random.choice(neg_cfg.long_texts)
            out_full = synth_any(text, k, prefix="long")
            if out_full is None: continue
            wav, sr = torchaudio.load(str(out_full))
            total_s = wav.shape[-1] / sr
            a, b = _rand_time_range(total_s, neg_cfg.clip_min_s, neg_cfg.clip_max_s, rng)
            out_clip = neg_dir / f"clip_{k:05d}.wav"
            _slice_wav(out_full, out_clip, a, min(b, total_s))
            created_neg.append(out_clip)

    meta = {
        "phrase": phrase,
        "lang": lang,
        "positives": [str(p) for p in created_pos],
        "negatives": [str(n) for n in created_neg],
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    out = generate_all(
        phrase="hey nova",
        lang="en",
        out_dir=Path("voice_samples/multi"),
        positives_per_engine=3,
        negatives_total=12,
        max_voices=1,
        seed=123,
    )
    print("Wrote:", json.dumps(out, indent=2))


