import os
from pathlib import Path
from .eleven_generator import generate_samples


def main():
    if not (os.getenv("ELEVEN_API_KEY") or os.getenv("ELEVENLABS_API_KEY")):
        print("Skipping Eleven test: ELEVEN_API_KEY not set.")
        return
    try:
        out = Path(__file__).parent / "samples" / "test"
        files = generate_samples(
            lang="en",
            count=2,
            phrase="hey nova",
            out_dir=out,
            resample_hz=16000,
            max_voices=2,
            seed=123,
            should_augment_punct=True,
        )
        print(f"Wrote {len(files)} files to {out}")
    except Exception as e:
        print("Eleven test failed/skipped:", e)


if __name__ == "__main__":
    main()


