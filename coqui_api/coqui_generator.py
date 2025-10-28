from __future__ import annotations
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import torchaudio  # pip install torchaudio
import torch

from .coqui_model_getter import (
    CoquiModelEntry,
    list_coqui_models,
    synthesize_to_wav,
)

import re
from text_augment import augment_punct


# Simple punctuation augmenter for short wake phrases
def augment_punct(
    phrase: str,
    rng: random.Random,
    *,
    p_replace_space: float = 0.6,    # chance to replace a space
    p_extra_end: float = 0.35,       # chance to add end punctuation
    candidates_space=(",", ";", ":", "—", "-"),
    candidates_end=("", ".", "!", "?", "…"),
) -> str:
    words = re.split(r"\s+", phrase.strip())
    if len(words) <= 1:
        end = rng.choice(candidates_end) if rng.random() < p_extra_end else ""
        return words[0] + end

    out = [words[0]]
    for w in words[1:]:
        sep = rng.choice(candidates_space) if rng.random() < p_replace_space else " "
        out.append(sep)
        out.append(w)

    text = "".join(out)
    if rng.random() < p_extra_end:
        text += rng.choice(candidates_end)
    return text


@dataclass
class Variability:
    speed: Tuple[float, float] = (0.95, 1.10)  # speech rate factor if supported by model
    speaker: Optional[str] = None             # for multi-speaker models
    language: Optional[str] = None            # for multilingual models


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _rand_in(rng: random.Random, lo_hi: Tuple[float, float]) -> float:
    lo, hi = lo_hi
    return rng.uniform(lo, hi)


def _safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:96]


def _manifest_path(root: Path) -> Path:
    return root / "models_manifest.json"


def _load_manifest(root: Path) -> dict:
    mp = _manifest_path(root)
    if mp.exists():
        try:
            return json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"models": {}, "updated": None}


def _save_manifest(root: Path, data: dict) -> None:
    data["updated"] = int(time.time())
    _manifest_path(root).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _quality_ok(wav_path: Path) -> bool:
    try:
        wav, sr = torchaudio.load(str(wav_path))
    except Exception:
        return False
    if wav.numel() == 0:
        return False
    wav = wav.mean(dim=0, keepdim=True)
    amp = wav.abs()
    # basic sanity: reasonable loudness, not mostly silence, some dynamic range
    rms = torch.sqrt((amp.pow(2).mean()).clamp(min=1e-12)).item()
    if not (0.003 <= rms <= 0.4):
        return False
    silence_prop = (amp < 1e-4).float().mean().item()
    if silence_prop > 0.6:
        return False
    lo = torch.quantile(amp, 0.05).item()
    hi = torch.quantile(amp, 0.95).item()
    if (hi - lo) < 0.02:
        return False
    return True


def generate_samples(
    *,
    lang: str,
    count: int,
    phrase: str,
    out_dir: str | Path,
    variability: Variability = Variability(),
    resample_hz: Optional[int] = 16000,
    models_dir: Path | str | None = None,   # manifest location; models are cached by TTS
    max_models: Optional[int] = None,        # legacy/internal name
    max_voices: Optional[int] = None,        # alias for Piper parity (limit number of voices/models)
    prefer_models: Optional[List[str]] = None,  # substrings to prioritize (e.g., ["ljspeech/vits"])
    avoid_models: Optional[List[str]] = None,   # substrings to exclude (e.g., ["blizzard2013/capacitron"])
    enable_speed: bool = False,                 # if True, vary speed; disabled by default for stability
    should_augment_punct: bool = True,          # control punctuation augmentation
    retries_per_sample: int = 2,                # re-synthesize if QC fails (try different speed/model)
    seed: Optional[int] = None,
    overwrite: bool = False,
) -> list[Path]:
    """
    Generate `count` samples per Coqui model for a given language.

    - Discovers Coqui models and downloads on demand via the TTS cache
    - Writes WAVs to `out_dir/<lang>/<dataset>/<model>/model_<idx>.wav`
    - Returns list of created file paths
    """
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Manifest lives under models_dir if provided, else under out_dir
    manifest_root = Path(models_dir) if models_dir is not None else out_dir
    _ensure_dir(manifest_root)
    manifest = _load_manifest(manifest_root)

    models = list_coqui_models(lang=lang)
    # Exclude unwanted models
    if avoid_models:
        am = tuple(avoid_models)
        models = [m for m in models if not any(p in m.model_name for p in am)]
    # Prefer certain models by ordering
    if prefer_models:
        pm = list(prefer_models)
        def score(m):
            for idx, pat in enumerate(pm):
                if pat in m.model_name:
                    return idx
            return len(pm) + 1
        models = sorted(models, key=lambda m: (score(m), m.model_name))
    # Respect either alias; prefer explicit max_voices if supplied
    limit = max_voices if (max_voices is not None) else max_models
    if limit is not None:
        models = models[:limit]
    if not models:
        raise RuntimeError(f"No Coqui models found for lang={lang!r}.")

    created: list[Path] = []

    for m_idx, m in enumerate(models):
        # Record in manifest
        manifest["models"][m.pretty_id] = {
            "lang": m.lang,
            "dataset": m.dataset,
            "model": m.model,
            "name": m.model_name,
        }

        # Folder per model
        m_folder = out_dir / m.lang / m.dataset / _safe_slug(m.model)
        _ensure_dir(m_folder)

        for i in range(count):
            phrase_aug = augment_punct(phrase, rng) if should_augment_punct else phrase
            # Slightly broader speed jitter while staying conservative
            speed = _rand_in(rng, (max(0.88, variability.speed[0]), min(1.10, variability.speed[1]))) if enable_speed else None
            speaker = variability.speaker
            language = variability.language
            text_tag = _safe_slug(phrase_aug.replace(" ", "_"))[:24]
            base_name = f"coqui_{_safe_slug(m.model)}_{i:04d}_{text_tag}"
            if speed is not None:
                base_name += f"_sp{speed:.2f}"
            out_wav = m_folder / f"{base_name}.wav"
            if out_wav.exists() and not overwrite:
                created.append(out_wav)
                continue

            # Attempt with QC and limited retries. Rotate models on retry for stability.
            ok = False
            for attempt in range(max(1, retries_per_sample)):
                use_m = m if attempt == 0 else models[(m_idx + attempt) % len(models)]
                use_speed = speed if attempt == 0 else None  # fall back to neutral speed
                tmp_out = out_wav.with_suffix(f".tmp.{attempt}.wav")
                synthesize_to_wav(
                    phrase_aug,
                    use_m,
                    tmp_out,
                    speed=use_speed,
                    speaker=speaker,
                    language=language,
                    resample_hz=resample_hz,
                )
                if _quality_ok(tmp_out):
                    tmp_out.rename(out_wav)
                    ok = True
                    break
                else:
                    tmp_out.unlink(missing_ok=True)
            if ok:
                created.append(out_wav)

    _save_manifest(manifest_root, manifest)
    return created


if __name__ == "__main__":
    files = generate_samples(
        lang="en",
        count=10,
        phrase="hey nova",
        out_dir=Path(__file__).parent / "samples",
        resample_hz=16000,
        max_models=1,  # quick demo
        seed=123,
    )
    print(f"Wrote {len(files)} files")


