from pathlib import Path
from .piper_generator import Variability, generate_samples


def main():
    out = Path(__file__).parent / "samples" / "test"
    files = generate_samples(
        lang="en",
        count=4,
        phrase="hey nova",
        out_dir=out,
        resample_hz=16000,
        max_voices=3,
        seed=123,
        variability=Variability(length_scale=(0.6, 1.10), noise_scale=(0.40, 0.80), noise_w_scale=(0.20, 1)),
    )
    print(f"Wrote {len(files)} files to {out}")


if __name__ == "__main__":
    main()


