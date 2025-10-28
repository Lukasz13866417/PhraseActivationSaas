from __future__ import annotations
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download  # pip install huggingface_hub
from piper.voice import PiperVoice, SynthesisConfig  # pip install piper-tts
import torchaudio  # pip install torchaudio

# ---- Repo constants ----
HF_REPO = "rhasspy/piper-voices"  # official Piper voices repository on the Hub


HERE = Path(__file__).parent
DEFAULT_VOICE_DIR = HERE / "voices"   # <piper_api>/voices

# Voice files live at:
#   <lang>/<locale>/<voice>/<quality>/<id>.onnx
#   <lang>/<locale>/<voice>/<quality>/<id>.onnx.json
# e.g. en/en_US/lessac/medium/en_US-lessac-medium.onnx (and .onnx.json)

VOICE_PATH_RE = re.compile(
    r"^(?P<lang>[a-z]{2})/(?P<locale>[a-z]{2}_[A-Z]{2})/(?P<voice>[a-z0-9_]+)"
    r"/(?P<quality>[^/]+)/(?P<id>[^/]+)\.onnx$"
)

@dataclass(frozen=True)
class VoiceEntry:
    lang: str          # e.g., "en"
    locale: str        # e.g., "en_US"
    voice: str         # e.g., "lessac"
    quality: str       # e.g., "medium"
    file_id: str       # e.g., "en_US-lessac-medium"
    onnx_path: str     # repo-relative path to .onnx
    json_path: str     # repo-relative path to .onnx.json

    @property
    def pretty_id(self) -> str:
        return f"{self.locale}/{self.voice}/{self.quality}"

def _iter_repo_onnx_paths(api: Optional[HfApi] = None) -> Iterable[str]:
    api = api or HfApi()
    # List all files in repo (tree on 'main') and filter to .onnx models
    # (You can also use snapshot_download to fetch everything, but listing keeps it light.)
    files = api.list_repo_files(repo_id=HF_REPO, repo_type="model", revision="main")
    for f in files:
        if f.endswith(".onnx"):
            yield f

def _parse_voice_path(p: str) -> Optional[VoiceEntry]:
    m = VOICE_PATH_RE.match(p)
    if not m:
        return None
    d = m.groupdict()
    onnx = p
    js = p + ".json"
    return VoiceEntry(
        lang=d["lang"],
        locale=d["locale"],
        voice=d["voice"],
        quality=d["quality"],
        file_id=d["id"],
        onnx_path=onnx,
        json_path=js,
    )

def list_piper_voices(
    lang: Optional[str] = None,
    locale: Optional[str] = None,
    quality: Optional[str] = None,
) -> List[VoiceEntry]:
    """
    Query the HF repo and return all matching VoiceEntry rows.
    Examples:
        list_piper_voices(lang="en")
        list_piper_voices(lang="pl", locale="pl_PL")
        list_piper_voices(lang="en", quality="medium")
    """
    api = HfApi()
    out: List[VoiceEntry] = []
    for path in _iter_repo_onnx_paths(api):
        v = _parse_voice_path(path)
        if not v:
            continue
        if lang and v.lang != lang:
            continue
        if locale and v.locale != locale:
            continue
        if quality and v.quality != quality:
            continue
        out.append(v)
    # stable order: by locale, then voice, then quality
    out.sort(key=lambda e: (e.locale, e.voice, e.quality))
    return out

def download_voice(
    v: VoiceEntry,
    local_dir: Path | str = DEFAULT_VOICE_DIR,
    revision: str = "main",
) -> Tuple[Path, Path]:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    onnx_local = hf_hub_download(
        repo_id=HF_REPO,
        filename=v.onnx_path,
        revision=revision,
        local_dir=local_dir,          # <- no local_dir_use_symlinks
    )
    json_local = hf_hub_download(
        repo_id=HF_REPO,
        filename=v.json_path,
        revision=revision,
        local_dir=local_dir,          # <- no local_dir_use_symlinks
    )
    return Path(onnx_local), Path(json_local)


def load_voice(
    v: VoiceEntry,
    local_dir: Path | str = DEFAULT_VOICE_DIR
) -> PiperVoice:
    onnx_fp, json_fp = download_voice(v, local_dir=local_dir)
    return PiperVoice.load(model_path=str(onnx_fp), config_path=str(json_fp))



def synthesize_to_wav(
    text: str,
    v: VoiceEntry,
    out_wav: Path | str,
    *,
    local_dir: Path | str = DEFAULT_VOICE_DIR,   # <- changed
    length_scale: float = 2.0,
    noise_scale: float = 3,
    noise_w_scale: float = 3,
    speaker_id: Optional[int] = None,
    resample_hz: Optional[int] = None,
) -> Path:
    voice = load_voice(v, local_dir=local_dir)
    cfg = SynthesisConfig(
        length_scale=length_scale,
        noise_scale=noise_scale,
        noise_w_scale=noise_w_scale,
        speaker_id=speaker_id,
    )

    out_wav = Path(out_wav)
    tmp_wav = out_wav if resample_hz is None else out_wav.with_suffix(".native.tmp.wav")

    # Stream chunks to WAV (AudioChunk exposes sample_rate/width/channels + int16 bytes)
    with wave.open(str(tmp_wav), "wb") as wf:
        first = True
        for chunk in voice.synthesize(text, cfg):
            if first:
                wf.setnchannels(chunk.sample_channels)  # typically 1
                wf.setsampwidth(chunk.sample_width)     # 2 bytes (PCM16)
                wf.setframerate(chunk.sample_rate)      # e.g., 22050
                first = False
            wf.writeframes(chunk.audio_int16_bytes)

    # Optional resample
    if resample_hz is not None:
        wav, sr = torchaudio.load(str(tmp_wav))
        if sr != resample_hz:
            wav = torchaudio.functional.resample(wav, sr, resample_hz)
        torchaudio.save(str(out_wav), wav, resample_hz)
        tmp_wav.unlink(missing_ok=True)

    return out_wav

# --------- Small CLI demo ---------
if __name__ == "__main__":
    # Example: list a few English and Polish voices and synthesize one sample each
    en_voices = list_piper_voices(lang="en")
    pl_voices = list_piper_voices(lang="pl", locale="pl_PL")

    print(f"Found {len(en_voices)} EN voices, {len(pl_voices)} PL voices.")
    if en_voices:
        print("EN example:", en_voices[0].pretty_id)
        synthesize_to_wav("hey nova", en_voices[0], "en_sample.wav", resample_hz=16000)
    if pl_voices:
        print("PL example:", pl_voices[0].pretty_id)
        synthesize_to_wav("hej nova", pl_voices[0], "pl_sample.wav", resample_hz=16000)
    print("Done.")
