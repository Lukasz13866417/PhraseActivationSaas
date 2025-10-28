Coqui TTS dataset generator

Usage example (similar to Piper):

```python
from pathlib import Path
from coqui_api.coqui_generator import generate_samples, Variability

files = generate_samples(
    lang="en",
    count=5,
    phrase="hey nova",
    out_dir=Path("voice_samples/positive"),
    resample_hz=16000,
    max_models=1,
    seed=42,
    variability=Variability(speed=(0.95, 1.10)),
)
print(len(files))
```

To list models without generating audio:

```python
from coqui_api.coqui_model_getter import list_coqui_models
print([m.pretty_id for m in list_coqui_models(lang="en")][:5])
```


