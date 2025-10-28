import torch

def simple_augment(wav: torch.Tensor, sr: int, g: torch.Generator) -> torch.Tensor:
    # 1) random gain (-6..+6 dB)
    gain_db = torch.randint(-6, 7, (1,), generator=g).item()
    wav = wav * (10.0 ** (gain_db / 20.0))

    # 2) small circular time shift (±80 ms)
    max_shift = int(0.08 * sr)
    if max_shift > 0:
        shift = torch.randint(-max_shift, max_shift + 1, (1,), generator=g).item()
        if shift != 0:
            wav = torch.roll(wav, shifts=shift, dims=-1)

    # 3) optional: add white-ish noise at random low SNR (10–20 dB) with 50% chance
    if torch.rand((), generator=g).item() < 0.5:
        snr_db = torch.randint(10, 21, (1,), generator=g).item()
        # ❗ use randn with explicit shape; randn_like doesn't take generator in your version
        noise = torch.randn(wav.shape, generator=g, device=wav.device, dtype=wav.dtype)
        p_sig = wav.pow(2).mean()
        p_noi = noise.pow(2).mean() + 1e-10
        scale = torch.sqrt((p_sig * (10 ** (-snr_db / 10))) / p_noi)
        wav = wav + scale * noise

    return torch.clamp(wav, -1.0, 1.0)