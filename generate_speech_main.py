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


# ------------- DB counting helpers -------------
def db_count_positives(con: sqlite3.Connection, phrase_norm: str) -> int:
    row = con.execute(
        "SELECT COUNT(*) FROM clips WHERE text_normalized = ?",
        (phrase_norm,),
    ).fetchone()
    return int(row[0]) if row else 0

def db_count_negatives_short(con: sqlite3.Connection, phrase_norm: str) -> int:
    # Identify short negatives by presence of clip_ms in params_json
    row = con.execute(
        """
        SELECT COUNT(*) FROM clips
        WHERE text_normalized != ?
          AND (params_json LIKE '%"clip_ms"%' )
        """,
        (phrase_norm,),
    ).fetchone()
    return int(row[0]) if row else 0

def db_count_negatives_long(con: sqlite3.Connection, phrase_norm: str) -> int:
    row = con.execute(
        """
        SELECT COUNT(*) FROM clips
        WHERE text_normalized != ?
          AND (params_json IS NULL OR params_json NOT LIKE '%"clip_ms"%')
        """,
        (phrase_norm,),
    ).fetchone()
    return int(row[0]) if row else 0

def db_counts_by_engine(con: sqlite3.Connection, phrase_norm: str) -> dict:
    """Return per-engine counts for pos, short, long, confuser."""
    out = {}
    # positives by engine
    for eng, cnt in con.execute(
        "SELECT engine, COUNT(*) FROM clips WHERE text_normalized=? GROUP BY engine",
        (phrase_norm,),
    ):
        out.setdefault(eng, {"pos": 0, "short": 0, "long": 0, "confuser": 0})
        out[eng]["pos"] = int(cnt)
    # negatives short
    for eng, cnt in con.execute(
        """
        SELECT engine, COUNT(*) FROM clips
        WHERE text_normalized != ? AND instr(params_json,'"clip_ms"')>0 AND instr(params_json,'"confuser"')=0
        GROUP BY engine
        """,
        (phrase_norm,),
    ):
        out.setdefault(eng, {"pos": 0, "short": 0, "long": 0, "confuser": 0})
        out[eng]["short"] = int(cnt)
    # negatives confuser (any length)
    for eng, cnt in con.execute(
        """
        SELECT engine, COUNT(*) FROM clips
        WHERE text_normalized != ? AND instr(params_json,'"confuser"')>0
        GROUP BY engine
        """,
        (phrase_norm,),
    ):
        out.setdefault(eng, {"pos": 0, "short": 0, "long": 0, "confuser": 0})
        out[eng]["confuser"] = int(cnt)
    # negatives long (non-short, non-confuser)
    for eng, cnt in con.execute(
        """
        SELECT engine, COUNT(*) FROM clips
        WHERE text_normalized != ?
          AND (params_json IS NULL OR instr(params_json,'"clip_ms"')=0)
          AND (params_json IS NULL OR instr(params_json,'"confuser"')=0)
        GROUP BY engine
        """,
        (phrase_norm,),
    ):
        out.setdefault(eng, {"pos": 0, "short": 0, "long": 0, "confuser": 0})
        out[eng]["long"] = int(cnt)
    return out


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

def _make_confuser_text(phrase: str, rng: random.Random) -> str:
    """Create a confuser similar to the phrase but WITHOUT containing it contiguously.
    Strategy: insert a filler token (not in phrase) between phrase tokens; fallback to small
    character perturbation. Re-check and repair to avoid contiguous match.
    """
    phrase_norm = norm_text(phrase)
    phrase_tokens = phrase_norm.split()
    # Avoid using any token from the phrase as filler (prevents accidental re-formation)
    candidate_fillers = ["uh", "please", "now", "okay", "and", "to", "kindly", "really"]
    fillers = [f for f in candidate_fillers if f not in phrase_tokens] or ["uh"]

    # Start from the original phrase tokens (preserve casing roughly)
    words = phrase.strip().split()

    # Prefer inserting a filler inside the phrase to break contiguity
    if len(words) >= 2:
        insert_pos = rng.randint(1, len(words) - 1)
        ins = rng.choice(fillers)
        words = words[:insert_pos] + [ins] + words[insert_pos:]
    else:
        # Single-word phrase: slight vowel perturbation or duplicate char
        w = list(words[0]) if words else []
        if w:
            k = rng.randint(0, len(w) - 1)
            w[k] = rng.choice([w[k], rng.choice("aeiou")])
            words = ["".join(w)]

    text = " ".join(words) if words else phrase

    # Re-check and repair if the normalized confuser still contains the phrase contiguously
    attempt = 0
    while phrase_norm in norm_text(text) and attempt < 3:
        attempt += 1
        # Try another filler at a different position
        if len(words) >= 2:
            insert_pos = rng.randint(1, len(words) - 1)
            ins = rng.choice(fillers)
            words = words[:insert_pos] + [ins] + words[insert_pos:]
        else:
            # Further perturb the single token
            w = list(words[0]) if words else []
            if w:
                k = rng.randint(0, len(w) - 1)
                w[k] = rng.choice([w[k], rng.choice("aeiou"), w[k] + w[k]])
                words = ["".join(w)]
        text = " ".join(words) if words else phrase

    # Final guaranteed break if still matched: mutate one phrase token deterministically
    if phrase_norm in norm_text(text) and words:
        # mutate last token slightly
        idx = max(0, len(words) - 1)
        words[idx] = words[idx] + rng.choice(["h", "a", "e"])
        text = " ".join(words)
    return text

def _sanitize_for_coqui(text: str) -> str:
    """Reduce risky punctuation/Unicode for Coqui TTS to avoid artifacts/segfaults.
    - Replace unicode dashes and unusual punctuation with spaces
    - Keep only letters/digits/space and , . ! ? ' characters
    - Collapse whitespace; cap length
    """
    # Normalize some unicode punctuation
    text = text.replace("—", " ").replace("–", " ").replace("‑", " ")
    text = text.replace(":", ", ")
    text = text.replace(";", ", ")
    text = text.replace("…", ".")
    text = text.replace("/", " ")
    # Remove other symbols
    text = re.sub(r"[^A-Za-z0-9 ,.!?'\n\r\t]", " ", text)
    # Collapse punctuation runs
    text = re.sub(r"[,.!?]{2,}", ". ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Cap overly long strings
    if len(text) > 240:
        text = text[:240].rsplit(" ", 1)[0]
    return text


# ------------- Orchestrator -------------
def _supports_internal_punct(engine: str, model_name: Optional[str]) -> bool:
    """Return True if engine/model safely supports internal punctuation like ! and ?.
    Conservative: allow for Piper/Eleven; allow for Coqui only on stable families.
    """
    if engine == "coqui":
        if not model_name:
            return False
        SAFE_FAMILIES = ("ljspeech/vits", "ljspeech/glow-tts", "vctk/vits")
        return any(f in model_name for f in SAFE_FAMILIES)
    # Piper and Eleven are robust with basic ! and ?
    return True
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
    "The fitnessgram pacer test is a 30 second test that measures your aerobic fitness.",
    "We will do what we must, but we do it for Aiur, not you.",
    "Do you seek knowledge of time travel?",
    "A wall of text is an excessively long post to a noticeboard or talk page discussion, which can often be so long that some don't read it.",
    "A text file that contains only text, with no formatting, is called plain text, and is the most basic way to store text.",
    "How to survive a wall of text? Read surrounding posts, or skim to determine whether the long post is largely substantive or mostly irrelevant. If it is the latter, apply trout and other remedies in suitable proportion. Simplest is just to ignore it if it's not relevant to you."
]

def generate_all(
    *,
    phrase: str,
    lang: str,
    out_dir: Path,
    db_path: Path,
    counts: Counts = Counts(),
    # Targets and extras
    target_pos: int = 100,
    target_short_negs: int = 200,
    target_long_negs: int = 200,
    target_confuser_negs: int = 200,
    extra_pos: int = 50,
    extra_short_negs: int = 50,
    extra_long_negs: int = 50,
    extra_confuser_negs: int = 50,
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
    print(f"[DBG] Piper voices: {len(piper_vs)}")
    if piper_vs:
        pv0 = piper_vs[0]
        print(f"[DBG] Piper example: {pv0.locale}/{pv0.voice}/{pv0.quality} ({pv0.file_id})")

    # Coqui setup
    coqui_ms: List[CoquiModelEntry] = []
    if HAS_COQUI:
        try:
            all_models = coqui_list_models(lang=lang)
            print(f"[DBG] Coqui models (raw): {len(all_models)}")
            # Avoid families known to be unstable in our environment
            UNSTABLE_COQUI_PATTERNS = ("capacitron", "neon", "tacotron", "tacotron2", "tortoise", "overflow")
            filtered = [m for m in all_models if not any(p in m.model_name or p in m.model for p in UNSTABLE_COQUI_PATTERNS)]
            PREFERRED_ORDER = ("ljspeech/vits", "ljspeech/glow-tts", "vctk/vits")
            def pref_score(model_name: str) -> int:
                for i, pat in enumerate(PREFERRED_ORDER):
                    if pat in model_name:
                        return i
                return len(PREFERRED_ORDER)
            filtered.sort(key=lambda m: (pref_score(m.model_name), m.model_name))
            # Keep only punctuation-safe families
            punct_safe = [m for m in filtered if _supports_internal_punct("coqui", m.model_name)]
            print(f"[DBG] Coqui models (punct-safe): {len(punct_safe)}")
            coqui_ms = punct_safe[:max_coqui_models] if max_coqui_models is not None else punct_safe
            print(f"[DBG] Coqui models (filtered): {len(coqui_ms)}")
            if coqui_ms:
                print(f"[DBG] Coqui example: {coqui_ms[0].model_name}")
        except Exception as e:
            print(f"[DBG] Coqui unavailable/skipped: {e}")
            coqui_ms = []

    # Eleven setup
    eleven_vs: List[ElevenVoiceEntry] = []
    if HAS_ELEVEN and (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")):
        try:
            eleven_vs = eleven_list_voices()
            if max_eleven_voices is not None:
                eleven_vs = eleven_vs[:max_eleven_voices]
            print(f"[DBG] Eleven voices: {len(eleven_vs)}")
            if eleven_vs:
                print(f"[DBG] Eleven example: {eleven_vs[0].name} ({eleven_vs[0].voice_id[:8]})")
        except Exception as e:
            print(f"[DBG] Eleven unavailable/skipped: {e}")
            eleven_vs = []
    else:
        print("[DBG] Eleven skipped (no API key or SDK not available)")

    # Unified pool (no positive/negative split in filesystem)
    pool_root = out_dir
    _ensure_dir(pool_root)

    # Compute current counts and required top-ups
    phrase_norm = norm_text(phrase)
    curr_pos = db_count_positives(con, phrase_norm)
    curr_short = db_count_negatives_short(con, phrase_norm)
    curr_long = db_count_negatives_long(con, phrase_norm)
    # Per-engine counts
    per_eng = db_counts_by_engine(con, phrase_norm)

    need_pos = max(0, target_pos - curr_pos) + extra_pos
    need_short = max(0, target_short_negs - curr_short) + extra_short_negs
    need_long = max(0, target_long_negs - curr_long) + extra_long_negs
    # Derive per-engine targets (even split)
    engines_available = [e for e in ("piper","coqui","eleven") if ((e=="piper" and piper_vs) or (e=="coqui" and coqui_ms) or (e=="eleven" and eleven_vs))]
    n_eng = max(1, len(engines_available))
    def split_even(total:int):
        base = total // n_eng
        rem = total - base*n_eng
        return [base + (1 if i<rem else 0) for i in range(n_eng)]
    tgt_pos_eng = dict(zip(engines_available, split_even(target_pos)))
    tgt_short_eng = dict(zip(engines_available, split_even(target_short_negs)))
    tgt_long_eng = dict(zip(engines_available, split_even(target_long_negs)))
    tgt_conf_eng = dict(zip(engines_available, split_even(target_confuser_negs)))
    extra_pos_eng = dict(zip(engines_available, split_even(extra_pos)))
    extra_short_eng = dict(zip(engines_available, split_even(extra_short_negs)))
    extra_long_eng = dict(zip(engines_available, split_even(extra_long_negs)))
    extra_conf_eng = dict(zip(engines_available, split_even(extra_confuser_negs)))

    # ---------- Positives (punct augmented) ----------
    def piper_positive(total_needed: int) -> int:
        if total_needed <= 0: return 0
        remaining = total_needed
        units = len(piper_vs) if piper_vs else 0
        if units == 0: return 0
        per = max(1, (remaining + units - 1) // units)
        produced = 0
        for v in piper_vs:
            if remaining <= 0: break
            voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
            model_id = v.file_id
            v_folder = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality
            _ensure_dir(v_folder)
            kmax = min(per, remaining)
            for _ in range(kmax):
                # Piper supports internal ! and ?
                text = augment_punct(
                    phrase,
                    rng,
                    p_replace_space=0.2,
                    candidates_space=(",", "!", "?"),
                    p_extra_end=0.3,
                    candidates_end=("", ".", "...", "!", "?"),
                )
                unique = f"{int(time.time()*1000)}_{rng.randint(0,9999):04d}"
                fname = f"piper_{_safe_slug(model_id)}_{unique}_{_safe_slug(text.replace(' ','_'))}.wav"
                out = v_folder / fname
                try:
                    piper_synth(
                        text,
                        v,
                        out,
                        resample_hz=resample_hz,
                        length_scale=rng.uniform(0.90, 1.10),
                        noise_scale=rng.uniform(0.40, 0.80),
                        noise_w_scale=rng.uniform(0.20, 0.60),
                        download_timeout=20.0,
                        download_retries=1,
                    )
                    db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=out, params={})
                    produced += 1
                    remaining -= 1
                except Exception as e:
                    print(f"[WARN] Piper positive failed for {voice_tag}: {e}")
        return produced

    def coqui_positive(total_needed: int) -> int:
        if total_needed <= 0 or not coqui_ms: return 0
        remaining = total_needed
        units = len(coqui_ms)
        per = max(1, (remaining + units - 1) // units)
        produced = 0
        for m in coqui_ms:
            if remaining <= 0: break
            model_id = m.model_name
            voice_tag = m.model
            m_folder = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model)
            _ensure_dir(m_folder)
            kmax = min(per, remaining)
            for _ in range(kmax):
                if _supports_internal_punct("coqui", m.model_name):
                    cand_space = (",", "!", "?")
                    cand_end = ("", ".", "...", "!", "?")
                else:
                    cand_space = (",",)
                    cand_end = ("", ".", "...")
                text = augment_punct(
                    phrase,
                    rng,
                    p_replace_space=0.2,
                    candidates_space=cand_space,
                    p_extra_end=0.3,
                    candidates_end=cand_end,
                )
                # sanitize for Coqui if needed
                text_coqui = _sanitize_for_coqui(text)
                unique = f"{int(time.time()*1000)}_{rng.randint(0,9999):04d}"
                fname = f"coqui_{_safe_slug(m.model)}_{unique}_{_safe_slug(text.replace(' ','_'))}.wav"
                out = m_folder / fname
                coqui_synth(text_coqui, m, out, resample_hz=resample_hz, speed=rng.uniform(0.90, 1.10))
                db_add_clip(con, engine="coqui", model=model_id, voice=voice_tag, lang=m.lang, text=text_coqui, path=out, params={})
                produced += 1
                remaining -= 1
        return produced

    def eleven_positive(total_needed: int) -> int:
        if total_needed <= 0 or not eleven_vs: return 0
        remaining = total_needed
        units = len(eleven_vs)
        per = max(1, (remaining + units - 1) // units)
        produced = 0
        for v in eleven_vs:
            if remaining <= 0: break
            model_id = "eleven_multilingual_v2"
            voice_tag = v.name or v.voice_id
            v_folder = pool_root / "eleven" / _safe_slug(voice_tag) / model_id
            _ensure_dir(v_folder)
            kmax = min(per, remaining)
            for _ in range(kmax):
                text = augment_punct(
                    phrase,
                    rng,
                    p_replace_space=0.2,
                    candidates_space=(",", "!", "?"),
                    p_extra_end=0.3,
                    candidates_end=("", ".", "...", "!", "?"),
                )
                unique = f"{int(time.time()*1000)}_{rng.randint(0,9999):04d}"
                fname = f"eleven_{_safe_slug(voice_tag)}_{_safe_slug(model_id)}_{unique}_{_safe_slug(text.replace(' ','_'))}.wav"
                out = v_folder / fname
                eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
                db_add_clip(con, engine="eleven", model=model_id, voice=voice_tag, lang=None, text=text, path=out, params={})
                produced += 1
                remaining -= 1
        return produced

    # Distribute positives per engine against per-engine targets
    produced = 0
    if piper_vs:
        cur = per_eng.get("piper", {}).get("pos", 0)
        need = max(0, tgt_pos_eng.get("piper", 0) - cur) + extra_pos_eng.get("piper", 0)
        produced += piper_positive(need)
    if coqui_ms:
        cur = per_eng.get("coqui", {}).get("pos", 0)
        need = max(0, tgt_pos_eng.get("coqui", 0) - cur) + extra_pos_eng.get("coqui", 0)
        produced += coqui_positive(need)
    if eleven_vs:
        cur = per_eng.get("eleven", {}).get("pos", 0)
        need = max(0, tgt_pos_eng.get("eleven", 0) - cur) + extra_pos_eng.get("eleven", 0)
        produced += eleven_positive(need)

    # ---------- Negatives: short (punct augmented, sliced 0.8–3.0s) ----------
    SHORT_MIN, SHORT_MAX = 0.8, 3.0
    def synth_any_engine(text: str, idx: int) -> Optional[Path]:
        # Cycle across available engines fairly
        available = []
        if piper_vs: available.append("piper")
        if coqui_ms: available.append("coqui")
        if eleven_vs: available.append("eleven")
        if not available:
            return None
        # pick engine by round-robin on idx
        eng = available[idx % len(available)]

        if eng == "piper":
            v = piper_vs[idx % len(piper_vs)]
            voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
            model_id = v.file_id
            out = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality / f"piper_{_safe_slug(model_id)}_{idx:05d}.wav"
            _ensure_dir(out.parent)
            try:
                piper_synth(
                    text, v, out, resample_hz=resample_hz,
                    length_scale=rng.uniform(0.95, 1.10),
                    noise_scale=rng.uniform(0.40, 0.80),
                    noise_w_scale=rng.uniform(0.20, 0.60),
                    download_timeout=20.0,
                    download_retries=1,
                )
                db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=out, params={})
                return out
            except Exception as e:
                print(f"[WARN] Piper short-neg failed for {voice_tag}: {e}")
                return None

        if eng == "coqui":
            m = coqui_ms[idx % len(coqui_ms)]
            out = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model) / f"coqui_{_safe_slug(m.model)}_{idx:05d}.wav"
            _ensure_dir(out.parent)
            coqui_synth(text, m, out, resample_hz=resample_hz, speed=rng.uniform(0.95, 1.10))
            db_add_clip(con, engine="coqui", model=m.model_name, voice=m.model, lang=m.lang, text=text, path=out, params={})
            return out

        if eng == "eleven":
            v = eleven_vs[idx % len(eleven_vs)]
            model_id = "eleven_multilingual_v2"
            out = pool_root / "eleven" / _safe_slug(v.name or v.voice_id) / model_id / f"eleven_{_safe_slug(v.name or v.voice_id)}_{model_id}_{idx:05d}.wav"
            _ensure_dir(out.parent)
            eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
            db_add_clip(con, engine="eleven", model=model_id, voice=(v.name or v.voice_id), lang=None, text=text, path=out, params={})
            return out

        return None

    texts_long = list(long_texts or DEFAULT_LONG_TEXTS)
    # Engine-specific short negatives (light punctuation augmentation)
    def gen_short_for_engine(eng: str, need: int):
        for k in range(need):
            text = augment_punct(
                rng.choice(texts_long),
                rng,
                p_replace_space=0.08,
                candidates_space=(",",),
                p_extra_end=0.15,
                candidates_end=("", "."),
            )
            if eng == "piper" and piper_vs:
                v = piper_vs[(k) % len(piper_vs)]
                voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
                model_id = v.file_id
                out = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality / f"piper_{_safe_slug(model_id)}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                try:
                    piper_synth(text, v, out, resample_hz=resample_hz,
                                length_scale=rng.uniform(0.95, 1.10),
                                noise_scale=rng.uniform(0.40, 0.80),
                                noise_w_scale=rng.uniform(0.20, 0.60),
                                download_timeout=20.0,
                                download_retries=1)
                    # slice
                    wav, sr = torchaudio.load(str(out))
                    total_s = wav.shape[-1] / sr
                    a, b = _rand_time_range(total_s, SHORT_MIN, SHORT_MAX, rng)
                    clip = out.with_name(out.stem + f"_clip_{int(a*1000)}_{int(b*1000)}_{rng.randint(0,9999):04d}.wav")
                    _slice_wav(out, clip, a, b)
                    db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=clip, params={"clip_ms": [int(a*1000), int(b*1000)]})
                except Exception as e:
                    print(f"[WARN] Piper short-neg failed for {voice_tag}: {e}")
            elif eng == "coqui" and coqui_ms:
                m = coqui_ms[(k) % len(coqui_ms)]
                out = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model) / f"coqui_{_safe_slug(m.model)}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                text_s = _sanitize_for_coqui(text)
                coqui_synth(text_s, m, out, resample_hz=resample_hz, speed=rng.uniform(0.95, 1.10))
                wav, sr = torchaudio.load(str(out))
                total_s = wav.shape[-1] / sr
                a, b = _rand_time_range(total_s, SHORT_MIN, SHORT_MAX, rng)
                clip = out.with_name(out.stem + f"_clip_{int(a*1000)}_{int(b*1000)}_{rng.randint(0,9999):04d}.wav")
                _slice_wav(out, clip, a, b)
                db_add_clip(con, engine="coqui", model=m.model_name, voice=m.model, lang=m.lang, text=text_s, path=clip, params={"clip_ms": [int(a*1000), int(b*1000)]})
            elif eng == "eleven" and eleven_vs:
                v = eleven_vs[(k) % len(eleven_vs)]
                model_id = "eleven_multilingual_v2"
                out = pool_root / "eleven" / _safe_slug(v.name or v.voice_id) / model_id / f"eleven_{_safe_slug(v.name or v.voice_id)}_{model_id}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
                wav, sr = torchaudio.load(str(out))
                total_s = wav.shape[-1] / sr
                a, b = _rand_time_range(total_s, SHORT_MIN, SHORT_MAX, rng)
                clip = out.with_name(out.stem + f"_clip_{int(a*1000)}_{int(b*1000)}_{rng.randint(0,9999):04d}.wav")
                _slice_wav(out, clip, a, b)
                db_add_clip(con, engine="eleven", model=model_id, voice=(v.name or v.voice_id), lang=None, text=text, path=clip, params={"clip_ms": [int(a*1000), int(b*1000)]})

    # per-engine needs for short
    if piper_vs:
        cur = per_eng.get("piper", {}).get("short", 0)
        need = max(0, tgt_short_eng.get("piper", 0) - cur) + extra_short_eng.get("piper", 0)
        gen_short_for_engine("piper", need)
    if coqui_ms:
        cur = per_eng.get("coqui", {}).get("short", 0)
        need = max(0, tgt_short_eng.get("coqui", 0) - cur) + extra_short_eng.get("coqui", 0)
        gen_short_for_engine("coqui", need)
    if eleven_vs:
        cur = per_eng.get("eleven", {}).get("short", 0)
        need = max(0, tgt_short_eng.get("eleven", 0) - cur) + extra_short_eng.get("eleven", 0)
        gen_short_for_engine("eleven", need)

    # ---------- Negatives: long (NO punctuation augmentation) ----------
    def gen_long_for_engine(eng: str, need: int):
        for k in range(need):
            text = rng.choice(texts_long)
            if eng == "piper" and piper_vs:
                v = piper_vs[(k) % len(piper_vs)]
                voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
                model_id = v.file_id
                out = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality / f"piper_{_safe_slug(model_id)}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                try:
                    piper_synth(text, v, out, resample_hz=resample_hz,
                                length_scale=rng.uniform(0.95, 1.10),
                                noise_scale=rng.uniform(0.40, 0.80),
                                noise_w_scale=rng.uniform(0.20, 0.60),
                                download_timeout=20.0,
                                download_retries=1)
                    db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=out, params={})
                except Exception as e:
                    print(f"[WARN] Piper long-neg failed for {voice_tag}: {e}")
            elif eng == "coqui" and coqui_ms:
                m = coqui_ms[(k) % len(coqui_ms)]
                out = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model) / f"coqui_{_safe_slug(m.model)}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                text_s = _sanitize_for_coqui(text)
                coqui_synth(text_s, m, out, resample_hz=resample_hz, speed=rng.uniform(0.95, 1.10))
                db_add_clip(con, engine="coqui", model=m.model_name, voice=m.model, lang=m.lang, text=text_s, path=out, params={})
            elif eng == "eleven" and eleven_vs:
                v = eleven_vs[(k) % len(eleven_vs)]
                model_id = "eleven_multilingual_v2"
                out = pool_root / "eleven" / _safe_slug(v.name or v.voice_id) / model_id / f"eleven_{_safe_slug(v.name or v.voice_id)}_{model_id}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
                db_add_clip(con, engine="eleven", model=model_id, voice=(v.name or v.voice_id), lang=None, text=text, path=out, params={})

    if piper_vs:
        cur = per_eng.get("piper", {}).get("long", 0)
        need = max(0, tgt_long_eng.get("piper", 0) - cur) + extra_long_eng.get("piper", 0)
        gen_long_for_engine("piper", need)
    if coqui_ms:
        cur = per_eng.get("coqui", {}).get("long", 0)
        need = max(0, tgt_long_eng.get("coqui", 0) - cur) + extra_long_eng.get("coqui", 0)
        gen_long_for_engine("coqui", need)
    if eleven_vs:
        cur = per_eng.get("eleven", {}).get("long", 0)
        need = max(0, tgt_long_eng.get("eleven", 0) - cur) + extra_long_eng.get("eleven", 0)
        gen_long_for_engine("eleven", need)

    # ---------- Negatives: confusers (mark in params) ----------
    def gen_confuser_for_engine(eng: str, need: int):
        for k in range(need):
            text = _make_confuser_text(phrase, rng)
            if eng == "piper" and piper_vs:
                v = piper_vs[(k) % len(piper_vs)]
                voice_tag = f"{v.locale}/{v.voice}/{v.quality}"
                model_id = v.file_id
                out = pool_root / "piper" / v.lang / v.locale / v.voice / v.quality / f"piper_{_safe_slug(model_id)}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                try:
                    piper_synth(text, v, out, resample_hz=resample_hz,
                                length_scale=rng.uniform(0.95, 1.10),
                                noise_scale=rng.uniform(0.40, 0.80),
                                noise_w_scale=rng.uniform(0.20, 0.60),
                                download_timeout=20.0,
                                download_retries=1)
                    db_add_clip(con, engine="piper", model=model_id, voice=voice_tag, lang=v.lang, text=text, path=out, params={"confuser": True})
                except Exception as e:
                    print(f"[WARN] Piper confuser failed for {voice_tag}: {e}")
            elif eng == "coqui" and coqui_ms:
                m = coqui_ms[(k) % len(coqui_ms)]
                out = pool_root / "coqui" / m.lang / m.dataset / _safe_slug(m.model) / f"coqui_{_safe_slug(m.model)}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                text_s = _sanitize_for_coqui(text)
                coqui_synth(text_s, m, out, resample_hz=resample_hz, speed=rng.uniform(0.95, 1.10))
                db_add_clip(con, engine="coqui", model=m.model_name, voice=m.model, lang=m.lang, text=text_s, path=out, params={"confuser": True})
            elif eng == "eleven" and eleven_vs:
                v = eleven_vs[(k) % len(eleven_vs)]
                model_id = "eleven_multilingual_v2"
                out = pool_root / "eleven" / _safe_slug(v.name or v.voice_id) / model_id / f"eleven_{_safe_slug(v.name or v.voice_id)}_{model_id}_{int(time.time()*1000)}_{rng.randint(0,9999):04d}.wav"
                _ensure_dir(out.parent)
                eleven_synth(text, v, out, model_id=model_id, resample_hz=resample_hz)
                db_add_clip(con, engine="eleven", model=model_id, voice=(v.name or v.voice_id), lang=None, text=text, path=out, params={"confuser": True})

    if piper_vs:
        cur = per_eng.get("piper", {}).get("confuser", 0)
        need = max(0, tgt_conf_eng.get("piper", 0) - cur) + extra_conf_eng.get("piper", 0)
        gen_confuser_for_engine("piper", need)
    if coqui_ms:
        cur = per_eng.get("coqui", {}).get("confuser", 0)
        need = max(0, tgt_conf_eng.get("coqui", 0) - cur) + extra_conf_eng.get("coqui", 0)
        gen_confuser_for_engine("coqui", need)
    if eleven_vs:
        cur = per_eng.get("eleven", {}).get("confuser", 0)
        need = max(0, tgt_conf_eng.get("eleven", 0) - cur) + extra_conf_eng.get("eleven", 0)
        gen_confuser_for_engine("eleven", need)

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
    # Targets and extras
    ap.add_argument("--target-pos", type=int, default=2000)
    ap.add_argument("--target-short-negs", type=int, default=6000)
    ap.add_argument("--target-long-negs", type=int, default=6000)
    ap.add_argument("--extra-pos", type=int, default=50)
    ap.add_argument("--extra-short-negs", type=int, default=50)
    ap.add_argument("--extra-long-negs", type=int, default=50)
    ap.add_argument("--max-piper-voices", type=int, default=2)
    ap.add_argument("--max-coqui-models", type=int, default=2)
    ap.add_argument("--max-eleven-voices", type=int, default=2)
    args = ap.parse_args()
    # Reduce thread contention that may cause segfaults in some libs
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        import torch  # type: ignore
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass
    except Exception:
        pass

    generate_all(
        phrase=args.phrase,
        lang=args.lang,
        out_dir=Path(args.out_dir),
        db_path=Path(args.db),
        target_pos=args.target_pos,
        target_short_negs=args.target_short_negs,
        target_long_negs=args.target_long_negs,
        extra_pos=args.extra_pos,
        extra_short_negs=args.extra_short_negs,
        extra_long_negs=args.extra_long_negs,
        resample_hz=args.resample_hz,
        max_piper_voices=args.max_piper_voices,
        max_coqui_models=args.max_coqui_models,
        max_eleven_voices=args.max_eleven_voices,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()