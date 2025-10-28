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
    

def make_df(dataset_folder_path: str) -> pd.DataFrame:
    pos_dir = Path(dataset_folder_path) / "positive"
    neg_dir = Path(dataset_folder_path) / "negative"

    pos_files = sorted([p for p in pos_dir.iterdir() if p.is_file()])
    neg_files = sorted([p for p in neg_dir.iterdir() if p.is_file()])

    data_pos = [[str(p), 1] for p in pos_files]
    data_neg = [[str(p), 0] for p in neg_files]

    df = pd.DataFrame(data_pos + data_neg, columns=["path", "label"])
    df["label"] = df["label"].astype(int)
    df = df.sort_values(["label", "path"], kind="mergesort", ignore_index=True)
    return df

