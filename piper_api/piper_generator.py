# piper_generator.py
from __future__ import annotations
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
import wave 

import torchaudio  # pip install torchaudio
from piper.voice import PiperVoice, SynthesisConfig  # pip install piper-tts

# Local helper module
from .piper_model_getter import (
    list_piper_voices,
    download_voice,
    load_voice,
    DEFAULT_VOICE_DIR,
)

import random
import re
from text_augment import augment_punct

# Simple punctuation augmenter for short wake phrases
def augment_punct(phrase: str, rng: random.Random,
                  *,
                  p_replace_space: float = 0.6,    # chance to replace a space
                  p_extra_end: float = 0.35,       # chance to add end punctuation
                  candidates_space = (",", ";", ":", "—", "-"),
                  candidates_end   = ("", ".", "!", "?", "…")) -> str:
    # normalize whitespace
    words = re.split(r"\s+", phrase.strip())
    if len(words) <= 1:
        # nothing to join; maybe just add end punctuation
        end = rng.choice(candidates_end) if rng.random() < p_extra_end else ""
        return words[0] + end

    out = [words[0]]
    for w in words[1:]:
        if rng.random() < p_replace_space:
            sep = rng.choice(candidates_space)
        else:
            sep = " "
        out.append(sep)
        out.append(w)

    text = "".join(out)
    if rng.random() < p_extra_end:
        text += rng.choice(candidates_end)
    return text

# ---------------- Config dataclasses ----------------

@dataclass
class Variability:
    length_scale: Tuple[float, float] = (0.95, 1.10)   # speech rate (↑ slower)
    noise_scale: Tuple[float, float]  = (0.40, 0.80)   # prosody variation
    noise_w_scale: Tuple[float, float]= (0.20, 0.60)   # cadence variation
    # If a voice is multi-speaker, you can pick a fixed speaker_id or a range (start, end)
    speaker_id: Optional[int] = None                   # None = default speaker


# ---------------- Internal utils ----------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _rand_in(rng: random.Random, lo_hi: Tuple[float, float]) -> float:
    lo, hi = lo_hi
    return rng.uniform(lo, hi)

def _safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:96]

def _manifest_path(root: Path) -> Path:
    return root / "voices_manifest.json"

def _load_manifest(root: Path) -> dict:
    mp = _manifest_path(root)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"voices": {}, "updated": None}

def _save_manifest(root: Path, data: dict) -> None:
    data["updated"] = int(time.time())
    _manifest_path(root).write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------- Public API ----------------

def generate_samples(
    *,
    lang: str,
    count: int,
    phrase: str,
    out_dir: str | Path,
    # Optional knobs
    variability: Variability = Variability(),
    resample_hz: Optional[int] = 16000,
    voices_dir: Path | str = DEFAULT_VOICE_DIR,  # where models/configs live (persist across runs)
    max_voices: Optional[int] = None,            # cap number of voices for quick runs
    seed: Optional[int] = None,
    overwrite: bool = False,                     # re-generate files if they exist
    should_augment_punct: bool = True,
    locale: Optional[str] = None,                # e.g., "pl_PL" or "en_US"
    quality: Optional[str] = None,               # e.g., "medium", "high"
) -> list[Path]:
    """
    Generate `count` samples per Piper voice in the given language.

    - Downloads missing voices (persisted in `voices_dir`)
    - Writes WAVs to `out_dir/<lang>/<voice_id>/voice_<idx>.wav`
    - Returns list of created file paths
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    voices_dir = Path(voices_dir)
    _ensure_dir(out_dir)
    _ensure_dir(voices_dir)

    # Persisted list of voices we’ve touched (for human-inspection; not required for correctness)
    manifest = _load_manifest(voices_dir)

    # Discover voices for the language (optionally restrict locale/quality)
    voices = list_piper_voices(lang=lang, locale=locale, quality=quality)
    if max_voices is not None:
        voices = voices[:max_voices]
    if not voices:
        raise RuntimeError(f"No Piper voices found for lang={lang!r} (locale={locale!r}, quality={quality!r}).")

    created: list[Path] = []

    for v in voices:
        # Ensure voice files are present on disk (download if missing)
        onnx_fp, json_fp = download_voice(v, local_dir=voices_dir)
        # Record in manifest
        manifest["voices"][v.pretty_id] = {
            "lang": v.lang,
            "locale": v.locale,
            "voice": v.voice,
            "quality": v.quality,
            "onnx": str(onnx_fp),
            "json": str(json_fp),
        }

        # Load once per voice
        voice = load_voice(v, local_dir=voices_dir)

        # Output folder per voice
        v_folder = out_dir / v.lang / v.locale / v.voice / v.quality
        _ensure_dir(v_folder)

        for i in range(count):
            # 1) text augmentation (punctuation/pauses + end-punct)
            phrase_aug = augment_punct(phrase, rng) if should_augment_punct else phrase

            # 2) sample synthesis knobs
            length_scale = _rand_in(rng, variability.length_scale)
            noise_scale  = _rand_in(rng, variability.noise_scale)
            noise_w      = _rand_in(rng, variability.noise_w_scale)
            speaker_id   = variability.speaker_id

            # 3) filename (include a short slug of the augmented text to make duplicates obvious)
            text_tag = _safe_slug(phrase_aug.replace(" ", "_"))[:24]
            name = f"piper_{_safe_slug(v.file_id)}_{i:04d}_{text_tag}_ls{length_scale:.2f}_n{noise_scale:.2f}_nw{noise_w:.2f}.wav"
            out_wav = v_folder / name
            if out_wav.exists() and not overwrite:
                created.append(out_wav); continue

            # 4) synthesize with Piper 1.3+ iterator API
            tmp_wav = out_wav.with_suffix(".tmp.native.wav")
            with wave.open(str(tmp_wav), "wb") as wf:
                first = True
                cfg = SynthesisConfig(
                    length_scale=length_scale,
                    noise_scale=noise_scale,
                    noise_w_scale=noise_w,
                    speaker_id=speaker_id,
                )
                for chunk in voice.synthesize(phrase_aug, cfg):  # <-- use augmented text
                    if first:
                        wf.setnchannels(chunk.sample_channels)
                        wf.setsampwidth(chunk.sample_width)
                        wf.setframerate(chunk.sample_rate)
                        first = False
                    wf.writeframes(chunk.audio_int16_bytes)

            # Optional resample → final file
            if resample_hz is not None:
                wav, sr = torchaudio.load(str(tmp_wav))
                if sr != resample_hz:
                    wav = torchaudio.functional.resample(wav, sr, resample_hz)
                torchaudio.save(str(out_wav), wav, resample_hz)
                Path(tmp_wav).unlink(missing_ok=True)
            else:
                tmp_wav.rename(out_wav)

            created.append(out_wav)

    # Persist manifest
    _save_manifest(voices_dir, manifest)
    return created


# ------------- tiny CLI -------------
if __name__ == "__main__":
    # Example usage: generate 3 samples per English voice
    files = generate_samples(
        lang="en",
        count=10,
        phrase="hey nova",
        out_dir=Path(__file__).parent / "samples",
        resample_hz=16000,
        max_voices=1,        # cap for quick demo
        seed=123,
    )
    print(f"Wrote {len(files)} files")
