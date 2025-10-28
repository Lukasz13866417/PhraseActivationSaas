#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sqlite3, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, List, Tuple

import torchaudio

# Engines
from text_augment import augment_punct

# Piper
from piper_api.piper_model_getter import (
    list_piper_voices as piper_list_voices,
    synthesize_to_wav as piper_synth,
    VoiceEntry as PiperVoiceEntry,
)
# Coqui
try:
    from coqui_api.coqui_model_getter import (
        list_coqui_models as coqui_list_models,
        synthesize_to_wav as coqui_synth,
        CoquiModelEntry,
    )
    HAS_COQUI = True
except Exception:
    HAS_COQUI = False

# ElevenLabs
try:
    from eleven_api.eleven_model_getter import (
        list_eleven_voices as eleven_list_voices,
        synthesize_to_wav as eleven_synth,
        ElevenVoiceEntry,
    )
    HAS_ELEVEN = True
except Exception:
    HAS_ELEVEN = False


# ------------- DB helpers -------------
def norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def file_sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY,
  engine TEXT NOT NULL,
  model TEXT NOT NULL,
  voice TEXT,
  lang TEXT,
  text_original TEXT NOT NULL,
  text_normalized TEXT NOT NULL,
  path TEXT NOT NULL UNIQUE,
  sample_rate INTEGER,
  duration_s REAL,
  hash_sha1 TEXT,
  params_json TEXT,
  created_at INTEGER DEFAULT (strftime('%s','now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS clips_fts
USING fts5(text_normalized, content='clips', content_rowid='id');

CREATE TRIGGER IF NOT EXISTS clips_ai AFTER INSERT ON clips
BEGIN
  INSERT INTO clips_fts(rowid, text_normalized) VALUES (new.id, new.text_normalized);
END;

CREATE TRIGGER IF NOT EXISTS clips_ad AFTER DELETE ON clips
BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, text_normalized) VALUES ('delete', old.id, old.text_normalized);
END;

CREATE TRIGGER IF NOT EXISTS clips_au AFTER UPDATE ON clips
BEGIN
  INSERT INTO clips_fts(clips_fts, rowid, text_normalized) VALUES ('delete', old.id, old.text_normalized);
  INSERT INTO clips_fts(rowid, text_normalized) VALUES (new.id, new.text_normalized);
END;
"""

def db_connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA_SQL)
    return con

def db_exists_text(con: sqlite3.Connection, *, engine: str, model: str, voice: str, text_norm: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM clips WHERE engine=? AND model=? AND voice=? AND text_normalized=? LIMIT 1",
        (engine, model, voice, text_norm),
    ).fetchone()
    return row is not None

def db_add_clip(
    con: sqlite3.Connection,
    *,
    engine: str,
    model: str,
    voice: str,
    lang: Optional[str],
    text: str,
    path: Path,
    params: dict,
):
    try:
        info = torchaudio.info(str(path))
        sr = info.sample_rate
        dur = info.num_frames / max(1, sr)
    except Exception:
        sr, dur = None, None
    sha1 = file_sha1(path)
    con.execute(
        """INSERT OR IGNORE INTO clips
           (engine,model,voice,lang,text_original,text_normalized,path,sample_rate,duration_s,hash_sha1,params_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (engine, model, voice, lang or None, text, norm_text(text), str(path), sr, dur, sha1, json.dumps(params)),
    )
    con.commit()


# ------------- Gen helpers -------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safe_slug(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)[:96]

def _rand_time_range(total_s: float, min_len: float, max_len: float, rng: random.Random) -> Tuple[float, float]:
    length = rng.uniform(min_len, max_len)
    if length >= total_s:
        return 0.0, max(0.0, total_s)
    start = rng.uniform(0.0, max(0.0, total_s - length))
    return start, start + length

def _slice_wav(in_wav: Path, out_wav: Path, start_s: float, end_s: float) -> None:
    wav, sr = torchaudio.load(str(in_wav))
    a = int(max(0, start_s) * sr)
    b = int(max(a + 1, end_s * sr))
    torchaudio.save(str(out_wav), wav[..., a:b], sr)


# ------------- Orchestrator -------------
@dataclass
class Counts:
    positives_per_voice: int = 2
    short_negs_total: int = 6
    long_negs_total: int = 6

DEFAULT_LONG_TEXTS = [
    "Please follow the safety protocols and keep your distance from the machinery.",
    "In the event of an emergency, proceed to the nearest exit in an orderly fashion.",
    "Our service hours are from nine A.M. to six P.M. Monday through Friday.",
    "For assistance, contact the help desk or check the online documentation portal.",
    "The quick brown fox jumps over the lazy dog.",
    "This is not a drill, this is not a drill, this is not a drill.",
    "How to survive a wall of text? Skim first, then focus if relevant.",
]

def generate_all(
    *,
    phrase: str,
    lang: str,
    out_dir: Path,
    db_path: Path,
    counts: Counts = Counts(),
    resample_hz: int = 16000,
    max_piper_voices: int = 2,
    max_coqui_models: int = 2,
    max_eleven_voices: int = 2,
    seed: Optional[int] = None,
    long_texts: Optional[Sequence[str]] = None,
) -> None:
    rng = random.Random(seed)
    out_dir = out_dir.resolve()
    _ensure_dir(out_dir)
    con = db_connect(db_path)

    # Piper setup
    piper_vs: List[PiperVoiceEntry] = piper_list_voices(lang=lang)
    if max_piper_voices is not None:
        piper_vs = piper_vs[:max_piper_voices]

    # Coqui setup
    coqui_ms: List[CoquiModelEntry] = []
    if HAS_COQUI:
        try:
            coqui_ms = coqui_list_models(lang=lang)
            if max_coqui_models is not None:
                coqui_ms = coqui_ms[:max_coqui_models]
        except Exception:
            coqui_ms = []

    # Eleven setup
    eleven_vs: List[ElevenVoiceEntry] = []
    if HAS_ELEVEN and (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")):
        try:
            eleven_vs = eleven_list_voices()
            if max_eleven_voices is not None:
                eleven_vs = eleven_vs[:max_eleven_voices]
        except Exception:
            eleven_vs = []

    # Unified pool (no positive/negative split in filesystem)
    pool_root = out_dir
    _ensure_dir(pool_root)

    # ---------- Positives (punct augmented) ----------
    def piper_positive():
        for v in piper_vs:
            voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
            model_id = v.file_id
            v_folder = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality
            _ensure_dir(v_folder)
            for i in range(counts.positives_per_voice):
                text = augment_punct(phrase, rng)
                if db_exists_text(con, engine="piper", model=model_id, voice=voice_tag, text_norm=norm_text(text)):
                    continue
                fname = f"piper_{_safe_slug(model_id)}_{i:04d}_{_safe_slug(text.replace(' ','_'))}.wav"
                out = v_folder / fname
                piper_synth(
                    text,
                    v,
                    out,
                    resample_hz=resample_hz,
                    length_scale=rng.uniform(0.90, 1.10),
                    noise_scale=rng.uniform(0.40, 0.80),
                    noise_w_scale=rng.uniform(0.20, 0.60),
                )
                db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=out, params={
                    "length_scale": float(re.search(r"_ls([0-9.]+)", fname).group(1)) if "_ls" in fname else None
                })

    def coqui_positive():
        for m in coqui_ms:
            model_id = m.model_name
            voice_tag = m.model
            m_folder = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model)
            _ensure_dir(m_folder)
            for i in range(counts.positives_per_voice):
                text = augment_punct(phrase, rng)
                if db_exists_text(con, engine="coqui", model=model_id, voice=voice_tag, text_norm=norm_text(text)):
                    continue
                fname = f"coqui_{_safe_slug(m.model)}_{i:04d}_{_safe_slug(text.replace(' ','_'))}.wav"
                out = m_folder / fname
                coqui_synth(text, m, out, resample_hz=resample_hz, speed=rng.uniform(0.90, 1.10))
                db_add_clip(con, engine="coqui", model=model_id, voice=voice_tag, lang=m.lang, text=text, path=out, params={})

    def eleven_positive():
        for v in eleven_vs:
            model_id = "eleven_multilingual_v2"
            voice_tag = v.name or v.voice_id
            v_folder = pool_root / "eleven" / _safe_slug(voice_tag) / model_id
            _ensure_dir(v_folder)
            for i in range(counts.positives_per_voice):
                text = augment_punct(phrase, rng)
                if db_exists_text(con, engine="eleven", model=model_id, voice=voice_tag, text_norm=norm_text(text)):
                    continue
                fname = f"eleven_{_safe_slug(voice_tag)}_{_safe_slug(model_id)}_{i:04d}_{_safe_slug(text.replace(' ','_'))}.wav"
                out = v_folder / fname
                eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
                db_add_clip(con, engine="eleven", model=model_id, voice=voice_tag, lang=None, text=text, path=out, params={})

    piper_positive()
    coqui_positive()
    eleven_positive()

    # ---------- Negatives: short (punct augmented, sliced 0.8–3.0s) ----------
    SHORT_MIN, SHORT_MAX = 0.8, 3.0
    def synth_any_engine(text: str, idx: int) -> Optional[Path]:
        sel = idx % 3
        # Piper
        if piper_vs:
            v = piper_vs[idx % len(piper_vs)]
            voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
            model_id = v.file_id
            out = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality / f"piper_{_safe_slug(model_id)}_{idx:05d}.wav"
            _ensure_dir(out.parent)
            piper_synth(
                text, v, out, resample_hz=resample_hz,
                length_scale=rng.uniform(0.95, 1.10),
                noise_scale=rng.uniform(0.40, 0.80),
                noise_w_scale=rng.uniform(0.20, 0.60),
            )
            db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=out, params={})
            return out
        # Coqui
        if coqui_ms:
            m = coqui_ms[idx % len(coqui_ms)]
            out = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model) / f"coqui_{_safe_slug(m.model)}_{idx:05d}.wav"
            _ensure_dir(out.parent)
            coqui_synth(text, m, out, resample_hz=resample_hz, speed=rng.uniform(0.95, 1.10))
            db_add_clip(con, engine="coqui", model=m.model_name, voice=m.model, lang=m.lang, text=text, path=out, params={})
            return out
        # Eleven
        if eleven_vs:
            v = eleven_vs[idx % len(eleven_vs)]
            model_id = "eleven_multilingual_v2"
            out = pool_root / "eleven" / _safe_slug(v.name or v.voice_id) / model_id / f"eleven_{_safe_slug(v.name or v.voice_id)}_{model_id}_{idx:05d}.wav"
            _ensure_dir(out.parent)
            eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
            db_add_clip(con, engine="eleven", model=model_id, voice=(v.name or v.voice_id), lang=None, text=text, path=out, params={})
            return out
        return None

    texts_long = list(long_texts or DEFAULT_LONG_TEXTS)
    for k in range(counts.short_negs_total):
        text = augment_punct(rng.choice(texts_long), rng)  # augmented
        full = synth_any_engine(text, k)
        if not full: continue
        try:
            wav, sr = torchaudio.load(str(full))
            total_s = wav.shape[-1] / sr
            a, b = _rand_time_range(total_s, SHORT_MIN, SHORT_MAX, rng)
            clip = full.with_name(full.stem + f"_clip_{int(a*1000)}_{int(b*1000)}.wav")
            _ensure_dir(clip.parent)
            _slice_wav(full, clip, a, b)
            # Record clip segment as separate row with same text
            db_add_clip(con, engine=db_engine_from_path(clip), model=db_model_from_path(clip), voice=db_voice_from_path(clip), lang=None, text=text, path=clip, params={"clip_ms": [int(a*1000), int(b*1000)]})
        except Exception:
            pass

    # ---------- Negatives: long (NO punctuation augmentation) ----------
    for k in range(counts.long_negs_total):
        text = rng.choice(texts_long)  # no augmentation
        # Use round-robin engine selection
        idx = k
        full = synth_any_engine(text, idx)
        # already recorded in db_add_clip above

def db_engine_from_path(p: Path) -> str:
    # crude: infer engine from path segment
    parts = [s.lower() for s in p.parts]
    if "piper" in parts: return "piper"
    if "coqui" in parts: return "coqui"
    if "eleven" in parts: return "eleven"
    return "unknown"

def db_model_from_path(p: Path) -> str:
    # best-effort: read token after engine folder
    parts = p.parts
    try:
        if "piper" in parts:
            # .../piper/<lang>/<locale>/<voice>/<quality>/piper_<fileid>_...
            name = p.name
            m = re.search(r"piper_([^_]+)", name)
            return m.group(1) if m else "piper_model"
        if "coqui" in parts:
            name = p.name
            m = re.search(r"coqui_([^_]+)", name)
            return m.group(1) if m else "coqui_model"
        if "eleven" in parts:
            # we prefixed eleven_<voice>_<model>...
            name = p.name
            m = re.search(r"eleven_[^_]+_([^_]+)", name)
            return m.group(1) if m else "eleven_model"
    except Exception:
        pass
    return "model"

def db_voice_from_path(p: Path) -> str:
    # best-effort voice tag
    name = p.name
    m = re.search(r"piper_([^_]+)", name)
    if m: return m.group(1)
    m = re.search(r"coqui_([^_]+)", name)
    if m: return m.group(1)
    m = re.search(r"eleven_([^_]+)_", name)
    if m: return m.group(1)
    return "voice"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phrase", required=True)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out-dir", default="voice_samples/multi")
    ap.add_argument("--db", default="db/tts.sqlite")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--resample-hz", type=int, default=16000)
    ap.add_argument("--pos-per-voice", type=int, default=2)
    ap.add_argument("--short-negs", type=int, default=6)
    ap.add_argument("--long-negs", type=int, default=6)
    ap.add_argument("--max-piper-voices", type=int, default=2)
    ap.add_argument("--max-coqui-models", type=int, default=2)
    ap.add_argument("--max-eleven-voices", type=int, default=2)
    args = ap.parse_args()

    counts = Counts(
        positives_per_voice=args.pos_per_voice,
        short_negs_total=args.short_negs,
        long_negs_total=args.long_negs,
    )
    generate_all(
        phrase=args.phrase,
        lang=args.lang,
        out_dir=Path(args.out_dir),
        db_path=Path(args.db),
        counts=counts,
        resample_hz=args.resample_hz,
        max_piper_voices=args.max_piper_voices,
        max_coqui_models=args.max_coqui_models,
        max_eleven_voices=args.max_eleven_voices,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()