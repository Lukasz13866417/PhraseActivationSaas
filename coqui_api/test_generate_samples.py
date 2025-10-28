from pathlib import Path
from .coqui_generator import generate_samples


def main():
    try:
        out = Path(__file__).parent / "samples" / "test"
        files = generate_samples(
            lang="en",
            count=3,
            phrase="hey nova",
            out_dir=out,
            resample_hz=None,  # avoid resample while testing stability
            max_models=3,
            seed=123,
            prefer_models=["tts_models/en/ljspeech/vits", "tts_models/en/ljspeech/glow-tts", "blizzard2013/capacitron"],
            avoid_models=["tts_models/en/ljspeech/tacotron2-DDC"],
            enable_speed=True,
        )
        print(f"Wrote {len(files)} files to {out}")
    except Exception as e:
        print("Coqui test skipped:", e)


if __name__ == "__main__":
    main()


