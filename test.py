#!/usr/bin/env python3
import sys, torch, torchaudio, torch.nn as nn, torch.nn.functional as F

# ---------- Model ----------
class LogSumExpPool(nn.Module):
    def __init__(self, tau=0.5): 
        super().__init__(); self.tau = tau
    def forward(self, x):                 # x: (B,T) frame logits
        return self.tau * torch.logsumexp(x / self.tau, dim=1)

class TinyCRNN(nn.Module):
    def __init__(self, in_ch=1, n_mels=40, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (B,64,1,T)
        self.gru = nn.GRU(input_size=64, hidden_size=hidden, num_layers=1, batch_first=True)
        self.frame_head = nn.Linear(hidden, 1)
        self.temporal_pool = LogSumExpPool(tau=0.5)       # soft-max over time

    def forward(self, x):                  # x: (B,1,40,T)
        h = self.conv(x)                   # (B,64,40,T)
        h = self.freq_pool(h).squeeze(2)   # (B,64,T)
        h = h.permute(0,2,1)               # (B,T,64)
        y,_ = self.gru(h)                  # (B,T,H)
        logits_t = self.frame_head(y).squeeze(-1)   # (B,T)
        window_logit = self.temporal_pool(logits_t) # (B,)
        return window_logit, logits_t              # logits (not sigmoid)

# ---------- Frontend config ----------
class SpectrogramConfig:
    def __init__(self, **d): 
        self.__dict__.update(d)
    def make_transforms(self):
        melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
            n_mels=self.n_mels, f_min=self.fmin, f_max=self.fmax,
            power=2.0, normalized=False, center=True, pad_mode="reflect",
        )
        to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
        return melspec, to_db

# ---------- Utilities ----------
def load_checkpoint(path="kw_model.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg = SpectrogramConfig(**ckpt["cfg"])
    model = TinyCRNN(in_ch=1, n_mels=cfg.n_mels, hidden=128).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    threshold = ckpt.get("threshold", None)
    return model, cfg, threshold

def wav_to_tensor(path, cfg, target_frames=200, device="cpu"):
    wav, sr = torchaudio.load(path)                 # [C, T]
    if sr != cfg.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, cfg.sample_rate)
    wav = wav.mean(0, keepdim=True)                 # [1, T]
    melspec, to_db = cfg.make_transforms()
    mel = melspec(wav)                               # [1, n_mels, T']
    x = to_db(mel)
    x = (x - cfg.mean_db) / (cfg.std_db + 1e-8)     # normalize like training

    # crop/pad to training length (change 200 if you trained with a different T)
    T = x.shape[-1]
    if target_frames is not None:
        if T < target_frames:
            x = F.pad(x, (0, target_frames - T))
        elif T > target_frames:
            start = (T - target_frames) // 2
            x = x[..., start:start + target_frames]
    x = x.unsqueeze(0).to(device)                   # [B=1, 1, n_mels, T]
    return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, cfg, threshold = load_checkpoint("kw_model.pt", device=device)
    except Exception as e:
        print("Failed to load kw_model.pt:", e)
        sys.exit(1)

    # If you trained with a different number of frames than 200, change this:
    TARGET_FRAMES = 200

    print("Loaded model. Device:", device.type)
    if threshold is not None:
        print(f"Saved threshold found: {threshold:.3f}")
    print("Enter path to a .wav file (empty line to quit).")

    while True:
        try:
            path = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if path == "":
            break
        try:
            x = wav_to_tensor(path, cfg, target_frames=TARGET_FRAMES, device=device)
            with torch.no_grad():
                logit, _ = model(x)
                prob = torch.sigmoid(logit)[0].item()
            if threshold is not None:
                decision = "TRIGGER" if prob >= threshold else "no-trigger"
                print(f"{path}\n  prob={prob:.4f}  threshold={threshold:.3f}  -> {decision}")
            else:
                print(f"{path}\n  prob={prob:.4f}  (no saved threshold; compare across files or choose one)")
        except Exception as e:
            print(f"Error processing '{path}': {e}")

if __name__ == "__main__":
    main()
