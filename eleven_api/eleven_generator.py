#!/usr/bin/env python3
from __future__ import annotations
import os, io, re, time, random, pathlib
from typing import Iterable, List, Optional, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import wave
from pathlib import Path
from typing import Optional, List

# ----------------------- Helpers -----------------------

def write_wav_from_pcm(raw_pcm: bytes, path: str, sr: int = 16000, channels: int = 1, sampwidth: int = 2):
    """Wrap raw PCM16 mono into a WAV container (no normalization)."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)  # 2 bytes = 16-bit PCM
        wf.setframerate(sr)
        wf.writeframesraw(raw_pcm)

def add_silence_pcm(raw_pcm: bytes, sr: int, lead_ms: int, tail_ms: int, sampwidth: int = 2, channels: int = 1) -> bytes:
    """Prepend/append digital silence (all-zero PCM)."""
    bytes_per_sample = sampwidth * channels
    lead_samples = int(sr * (lead_ms / 1000.0))
    tail_samples = int(sr * (tail_ms / 1000.0))
    return (b"\x00" * (lead_samples * bytes_per_sample)) + raw_pcm + (b"\x00" * (tail_samples * bytes_per_sample))

def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s)).strip("-")[:n]

def normalize_text(s: str) -> str:
    """Lowercase + collapse spaces + strip basic punctuation for equality/containment checks."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------- Core TTS -----------------------

@dataclass(frozen=True)
class ProsodyRanges:
    speed: Tuple[float, float]      = (0.80, 1.15)
    stability: Tuple[float, float]  = (0.18, 0.82)
    style: Tuple[float, float]      = (0.00, 0.80)

@dataclass(frozen=True)
class SilenceRanges:
    lead_ms: Tuple[int, int] = (80, 1000)
    tail_ms: Tuple[int, int] = (20, 150)

PUNCT_TEMPLATES = [
    "{phrase}",
    "{phrase}!",
    "{phrase}?",
    "{w1}, {w2}",          # tiny pause/intonation
    "{w1} — {w2}",         # em dash
]

def pick_variant(text: str) -> str:
    w = text.strip().split()
    if len(w) == 2 and random.random() < 0.6:
        tpl = random.choice(PUNCT_TEMPLATES)
        return tpl.format(phrase=text, w1=w[0].capitalize(), w2=w[1])
    return text

def build_client() -> ElevenLabs:
    load_dotenv()
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ELEVEN_API_KEY in your environment (or .env).")
    return ElevenLabs(api_key=api_key)

# Lightweight bulk generation API (parity with Piper/Coqui)
from .eleven_model_getter import list_eleven_voices, synthesize_to_wav, ElevenVoiceEntry
from text_augment import augment_punct


def generate_samples(
    *,
    lang: str,                 # kept for parity; Eleven voices are not strictly lang-scoped
    count: int,
    phrase: str,
    out_dir: str | Path,
    resample_hz: Optional[int] = 16000,
    max_voices: Optional[int] = None,
    seed: Optional[int] = None,
    overwrite: bool = False,
    model_id: Optional[str] = None,
    should_augment_punct: bool = True,
) -> list[Path]:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        voices: List[ElevenVoiceEntry] = list_eleven_voices()
    except Exception as e:
        raise RuntimeError(f"ElevenLabs voices unavailable: {e}")
    if max_voices is not None:
        voices = voices[:max_voices]
    if not voices:
        raise RuntimeError("No ElevenLabs voices available.")

    created: list[Path] = []
    for v in voices:
        v_folder = out_dir / (v.name or "voice") / (model_id or "eleven_multilingual_v2")
        v_folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            phrase_aug = augment_punct(phrase, rng) if should_augment_punct else phrase
            text_tag = re.sub(r"\s+", "_", phrase_aug)[:24]
            out = v_folder / f"{text_tag}_{i:04d}.wav"
            if out.exists() and not overwrite:
                created.append(out); continue
            synthesize_to_wav(
                phrase_aug,
                v,
                out,
                model_id=model_id,
                resample_hz=resample_hz,
            )
            created.append(out)
    return created

 


