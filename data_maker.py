from dataclasses import dataclass
import torch, torchaudio, pandas as pd
from torch.utils.data import Dataset
from typing import Callable
from pathlib import Path

Augment = Callable[[torch.Tensor, int, torch.Generator], torch.Tensor]

import hashlib

def stable_seed_from(path: str, epoch: int) -> int:
    h = hashlib.md5(f"{path}|{epoch}".encode()).hexdigest()
    return int(h[:8], 16)  # 32-bit

def make_augment(epoch: int) -> Augment:
    def augment(wav: torch.Tensor, sr: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(epoch)  

        return wav
    return augment

@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7600.0
    mean_db: float = 0.0      # set your dataset stats later
    std_db: float = 1.0

    # Convenience: build the torchaudio transforms from this config
    def make_transforms(self):
        melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.fmin,
            f_max=self.fmax,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        return melspec, to_db

class KeywordDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg: SpectrogramConfig, augment: Augment):
        self.df = df
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.melspec, self.to_db = cfg.make_transforms()
        self.mean, self.std = cfg.mean_db, cfg.std_db
        self.augment = augment
        self.epoch = 0  
        self.p_reverse_neg = 0.15



    def set_epoch(self, e: int):  
        self.epoch = e

    def __len__(self): return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        path_str, y = str(row["path"]), int(row["label"])

        wav, sr = torchaudio.load(path_str)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(0, keepdim=True)

        seed = stable_seed_from(path_str, self.epoch)
        g = torch.Generator(device="cpu").manual_seed(seed)

        if (y == 0) and (self.p_reverse_neg > 0):
            if torch.rand((), generator=g).item() < self.p_reverse_neg:
                wav = torch.flip(wav, dims=[-1])

        if self.augment is not None:
            wav = self.augment(wav, self.sr, g)

        mel = self.melspec(wav)
        x = self.to_db(mel)
        x = (x - self.mean) / (self.std + 1e-8)
        return x.float(), torch.tensor(y, dtype=torch.long)
    

import sqlite3
import re
from typing import Optional, Sequence

def _norm_text_db(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def make_df_from_db(
    db_path: str,
    phrase: str,
    *,
    limit_pos: Optional[int] = None,
    limit_neg: Optional[int] = None,
    engines: Optional[Sequence[str]] = None,    # e.g. ["piper","coqui","eleven"]
    langs: Optional[Sequence[str]] = None,      # e.g. ["en","pl"]
    min_dur_s: float = 0.25,
    max_dur_s: float = 10.0,
) -> pd.DataFrame:
    phrase_norm = _norm_text_db(phrase)
    con = sqlite3.connect(db_path)

    # Build optional filters
    where_extra = []
    params_pos = [phrase_norm, min_dur_s, max_dur_s]
    params_neg = [phrase_norm, min_dur_s, max_dur_s, phrase_norm]  # last is for FTS

    if engines:
        placeholders = ",".join(["?"] * len(engines))
        where_extra.append(f"engine IN ({placeholders})")
        params_pos.extend(engines)
        params_neg.extend(engines)
    if langs:
        placeholders = ",".join(["?"] * len(langs))
        where_extra.append(f"(lang IN ({placeholders}) OR lang IS NULL)")
        params_pos.extend(langs)
        params_neg.extend(langs)

    where_tail = (" AND " + " AND ".join(where_extra)) if where_extra else ""

    # Positives: exact normalized match
    q_pos = f"""
    SELECT path, 1 AS label
    FROM clips
    WHERE text_normalized = ?
      AND duration_s BETWEEN ? AND ?
      {where_tail}
    ORDER BY created_at DESC
    """
    if limit_pos:
        q_pos += f" LIMIT {int(limit_pos)}"

    pos = pd.read_sql_query(q_pos, con, params=params_pos)

    # Negatives: exclude any row containing the phrase tokens
    # Prefer FTS; fallback to LIKE if FTS not available
    try:
        q_neg = f"""
        SELECT path, 0 AS label
        FROM clips
        WHERE text_normalized != ?
          AND duration_s BETWEEN ? AND ?
          AND rowid NOT IN (
            SELECT rowid FROM clips_fts WHERE text_normalized MATCH ?
          )
          {where_tail}
        ORDER BY created_at DESC
        """
        if limit_neg:
            q_neg += f" LIMIT {int(limit_neg)}"
        neg = pd.read_sql_query(q_neg, con, params=params_neg)
    except Exception:
        like = f"%{phrase_norm}%"
        params_neg_like = [phrase_norm, min_dur_s, max_dur_s, like]
        if engines: params_neg_like.extend(engines)
        if langs: params_neg_like.extend(langs)
        q_neg_like = f"""
        SELECT path, 0 AS label
        FROM clips
        WHERE text_normalized != ?
          AND duration_s BETWEEN ? AND ?
          AND text_normalized NOT LIKE ?
          {where_tail}
        ORDER BY created_at DESC
        """
        if limit_neg:
            q_neg_like += f" LIMIT {int(limit_neg)}"
        neg = pd.read_sql_query(q_neg_like, con, params=params_neg_like)

    con.close()

    df = pd.concat([pos, neg], ignore_index=True)
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    return df