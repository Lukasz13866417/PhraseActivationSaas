from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torchaudio
from text_augment import augment_punct  # not used here, but keeping naming parity across modules

try:
    from elevenlabs import ElevenLabs, Voice
except Exception as e:
    ElevenLabs = None  # type: ignore
    Voice = None  # type: ignore
    _ELEVEN_IMPORT_ERROR = e
else:
    _ELEVEN_IMPORT_ERROR = None


@dataclass(frozen=True)
class ElevenVoiceEntry:
    voice_id: str
    name: str
    category: Optional[str]

    @property
    def pretty_id(self) -> str:
        return f"{self.name} ({self.voice_id[:8]})"


def _client_from_env() -> "ElevenLabs":
    if ElevenLabs is None:
        raise RuntimeError("elevenlabs SDK not installed: pip install elevenlabs") from _ELEVEN_IMPORT_ERROR
    api_key = os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("Set ELEVEN_API_KEY in environment to use ElevenLabs.")
    return ElevenLabs(api_key=api_key)


def list_eleven_voices() -> List[ElevenVoiceEntry]:
    client = _client_from_env()
    resp = client.voices.get_all()
    entries: List[ElevenVoiceEntry] = []
    for v in resp.voices:
        entries.append(ElevenVoiceEntry(voice_id=v.voice_id, name=v.name, category=getattr(v, "category", None)))
    entries.sort(key=lambda e: (e.category or "", e.name))
    return entries


def synthesize_to_wav(
    text: str,
    voice: ElevenVoiceEntry,
    out_wav: Path | str,
    *,
    model_id: Optional[str] = None,   # e.g. "eleven_multilingual_v2"
    stability: Optional[float] = None,
    similarity_boost: Optional[float] = None,
    style: Optional[float] = None,
    use_speaker_boost: Optional[bool] = None,
    resample_hz: Optional[int] = None,
) -> Path:
    client = _client_from_env()

    # Build optional voice settings if provided
    voice_settings = {}
    if stability is not None: voice_settings["stability"] = float(stability)
    if similarity_boost is not None: voice_settings["similarity_boost"] = float(similarity_boost)
    if style is not None: voice_settings["style"] = float(style)
    if use_speaker_boost is not None: voice_settings["use_speaker_boost"] = bool(use_speaker_boost)

    # Default model if not provided
    if model_id is None:
        model_id = "eleven_multilingual_v2"

    resp = client.text_to_speech.convert(
        voice_id=voice.voice_id,
        text=text,
        model_id=model_id,
        voice_settings=voice_settings or None,
        # Request raw PCM16 stream at 16 kHz, then wrap into WAV
        output_format="pcm_16000",
    )

    # Handle both bytes return and streaming (iterable of bytes)
    if isinstance(resp, (bytes, bytearray, memoryview)):
        audio_bytes = bytes(resp)
    else:
        # assume iterable of chunks
        chunks = []
        for chunk in resp:  # type: ignore[assignment]
            if chunk:
                chunks.append(chunk)
        audio_bytes = b"".join(chunks)

    out_wav = Path(out_wav)
    # Prefix filename with eleven_<voice>_<model>
    try:
        voice_tag = f"eleven_{_safe_slug(voice.name)}_{_safe_slug(model_id)}"
        if out_wav.name and not out_wav.name.startswith("eleven_"):
            out_wav = out_wav.with_name(f"{voice_tag}_{out_wav.name}")
    except Exception:
        pass

    tmp_wav = out_wav if (resample_hz is None or resample_hz == 16000) else out_wav.with_suffix(".native.tmp.wav")
    tmp_wav.parent.mkdir(parents=True, exist_ok=True)

    # Wrap PCM into a WAV container
    import wave
    with wave.open(str(tmp_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframesraw(audio_bytes)

    if resample_hz is not None and resample_hz != 16000:
        wav, sr = torchaudio.load(str(tmp_wav))
        if sr != resample_hz:
            wav = torchaudio.functional.resample(wav, sr, resample_hz)
        torchaudio.save(str(out_wav), wav, resample_hz)
        if tmp_wav != out_wav:
            Path(tmp_wav).unlink(missing_ok=True)

    return out_wav


