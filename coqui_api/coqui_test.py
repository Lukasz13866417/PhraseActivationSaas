from pathlib import Path
from .coqui_model_getter import list_coqui_models, synthesize_to_wav


def main():
    models = list_coqui_models(lang="en")
    if not models:
        print("No Coqui models found for en.")
        return
    # Prefer more stable English models for short phrases
    preferred = (
        "tts_models/en/ljspeech/vits",
        "tts_models/en/ljspeech/glow-tts",
        "tts_models/en/ljspeech/tacotron2-DDC",
    )
    name_to_model = {m.model_name: m for m in models}
    m = None
    for nm in preferred:
        if nm in name_to_model:
            m = name_to_model[nm]; break
    if m is None:
        m = models[0]
    print("Using:", m.model_name)
    out = Path(__file__).parent / "samples" / "en" / "test.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Disable resampling first to verify model output quality at native rate.
    synthesize_to_wav("hey nova", m, out, resample_hz=None)
    print("Wrote:", out)


if __name__ == "__main__":
    main()


