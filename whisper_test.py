import torch
from transformers import pipeline
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline(
    "automatic-speech-recognition",
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device
)

result = pipe("/home/roshni/Project/Whisper/audio_files/seedtts_ref_en_2.wav")
print("Transcription:", result["text"])
