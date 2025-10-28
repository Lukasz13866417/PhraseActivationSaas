from __future__ import annotations
import inspect
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torchaudio  # pip install torchaudio
import torch
import numpy as _np

# PyTorch 2.6+ defaults torch.load(weights_only=True), which restricts unpickling
# Allowlist numpy's scalar class commonly used in older checkpoints.
try:
    from numpy.core.multiarray import scalar as _np_scalar  # type: ignore
    try:
        torch.serialization.add_safe_globals([_np_scalar])  # type: ignore[attr-defined]
    except Exception:
        pass
except Exception:
    pass

# Also allowlist numpy.dtype used in some checkpoints
try:
    torch.serialization.add_safe_globals([_np.dtype])  # type: ignore[attr-defined]
except Exception:
    pass

try:
    from TTS.api import TTS  # pip install TTS (not available on Python >=3.12)
except Exception as e:  # broad to catch ImportError and runtime import errors
    TTS = None  # type: ignore
    _TTS_IMPORT_ERROR = e
else:
    _TTS_IMPORT_ERROR = None

try:
    # Prefer ModelManager for listing models (works across TTS versions)
    from TTS.utils.manage import ModelManager  # type: ignore
except Exception:
    ModelManager = None  # type: ignore


@dataclass(frozen=True)
class CoquiModelEntry:
    lang: str
    dataset: str
    model: str
    model_name: str

    @property
    def pretty_id(self) -> str:
        return f"{self.lang}/{self.dataset}/{self.model}"


def _list_available_model_names() -> List[str]:
    if TTS is None:
        raise RuntimeError(
            "Coqui TTS is unavailable. Install `TTS` on Python <3.12, or use a Python 3.10/3.11 environment."
        ) from _TTS_IMPORT_ERROR
    # First try the ModelManager API
    if ModelManager is not None:
        try:
            manager = ModelManager()
            names = manager.list_models()
            if names:
                return names
        except Exception:
            pass
    # Fallback: try class attribute (some versions expose TTS.list_models)
    try:
        maybe = getattr(TTS, "list_models", None)
        if callable(maybe):
            try:
                return maybe()  # may fail if it's an instance method in this version
            except TypeError:
                pass
    except Exception:
        pass
    raise RuntimeError("Unable to list Coqui models via available APIs.")


def list_coqui_models(lang: Optional[str] = None) -> List[CoquiModelEntry]:
    """
    Return available Coqui TTS models, optionally filtered by language code.
    Uses the built-in model registry from the TTS library.
    """
    names = [m for m in _list_available_model_names() if m.startswith("tts_models/")]
    entries: List[CoquiModelEntry] = []
    for name in names:
        parts = name.split("/")
        if len(parts) < 4:
            continue
        # parts: ["tts_models", lang, dataset, model (may contain "/" in future)]
        lang_code = parts[1]
        dataset = parts[2]
        model = "/".join(parts[3:])
        if lang and lang_code != lang:
            continue
        entries.append(CoquiModelEntry(lang=lang_code, dataset=dataset, model=model, model_name=name))
    entries.sort(key=lambda e: (e.lang, e.dataset, e.model))
    return entries


def load_tts(entry: CoquiModelEntry, *, gpu: bool = False):
    """Load a Coqui TTS model by registry name. Downloads on first use into cache."""
    if TTS is None:
        raise RuntimeError(
            "Coqui TTS is unavailable. Install `TTS` on Python <3.12, or use a Python 3.10/3.11 environment."
        ) from _TTS_IMPORT_ERROR
    return TTS(model_name=entry.model_name, progress_bar=False, gpu=gpu)


def _build_supported_kwargs(func, candidate_kwargs: dict) -> dict:
    """Return only kwargs supported by the callable's signature."""
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in candidate_kwargs.items() if (v is not None and k in allowed)}
    except (TypeError, ValueError):
        # If introspection fails, fall back to passing required-only args later
        return {}


def synthesize_to_wav(
    text: str,
    entry: CoquiModelEntry,
    out_wav: Path | str,
    *,
    speed: Optional[float] = None,
    speaker: Optional[str] = None,
    language: Optional[str] = None,
    resample_hz: Optional[int] = None,
    gpu: bool = False,
) -> Path:
    """
    Synthesize speech with a Coqui model into a WAV file.

    - Tries to pass optional args (speed/speaker/language) only if supported
    - Optionally resamples the output to a fixed sample rate (e.g., 16 kHz)
    """
    out_wav = Path(out_wav)
    tmp_wav = out_wav if resample_hz is None else out_wav.with_suffix(".native.tmp.wav")

    tts = load_tts(entry, gpu=gpu)

    base_kwargs = {
        "text": text,
        "file_path": str(tmp_wav),
    }
    opt_kwargs = _build_supported_kwargs(
        tts.tts_to_file,
        {
            "speed": speed,
            "speaker": speaker,
            "language": language,
        },
    )

    try:
        tts.tts_to_file(**{**base_kwargs, **opt_kwargs})
    except TypeError:
        # Retry with only required args if the wrapper rejected optional ones at runtime
        tts.tts_to_file(**base_kwargs)

    if resample_hz is not None:
        wav, sr = torchaudio.load(str(tmp_wav))
        if sr != resample_hz:
            wav = torchaudio.functional.resample(wav, sr, resample_hz)
        torchaudio.save(str(out_wav), wav, resample_hz)
        Path(tmp_wav).unlink(missing_ok=True)

    return out_wav


