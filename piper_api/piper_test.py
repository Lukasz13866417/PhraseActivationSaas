import wave
from pathlib import Path
from piper.voice import PiperVoice, SynthesisConfig

MODEL  = Path("en_US-lessac-medium.onnx")
CONFIG = Path("en_US-lessac-medium.onnx.json")

voice = PiperVoice.load(model_path=str(MODEL), config_path=str(CONFIG))

cfg = SynthesisConfig(
    length_scale=1.05,
    noise_scale=0.8,
    noise_w_scale=0.8,
)

text = "hey nova"

with wave.open("out.wav", "wb") as wf:
    first = True
    frames = 0
    for chunk in voice.synthesize(text, cfg):
        if first:
            wf.setnchannels(chunk.sample_channels)   # usually 1
            wf.setsampwidth(chunk.sample_width)      # 2 bytes (16-bit)
            wf.setframerate(chunk.sample_rate)       # e.g., 22050
            first = False
        b = chunk.audio_int16_bytes
        if b:
            wf.writeframes(b)
            frames += len(b) // chunk.sample_width
print("frames written:", frames)
