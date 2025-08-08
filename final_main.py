import os
import time
import tempfile
import threading
import hashlib
from dataclasses import dataclass, field
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import gradio as gr
import torch
from transformers import pipeline
from kokoro import KPipeline
from API_call import ask_umass_assistant
import xxhash

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing Whisper pipeline...")
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device
)
print("Whisper pipeline initialized!")

os.environ['ESPEAK_DATA_PATH'] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

response_cache = {}
tts_cache = {}
cache_lock = threading.Lock()

@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)
    stopped: bool = False
    model_outs: any = None
    is_first_interaction: bool = True

def get_cache_key(text):
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_response(text):
    cache_key = get_cache_key(text)
    with cache_lock:
        return response_cache.get(cache_key)

def cache_response(text, response):
    cache_key = get_cache_key(text)
    with cache_lock:
        response_cache[cache_key] = response
        if len(response_cache) > 1000:
            for key in list(response_cache.keys())[:100]:
                del response_cache[key]

def get_cached_audio(text):
    cache_key = get_cache_key(text)
    with cache_lock:
        return tts_cache.get(cache_key)

def cache_audio(text, audio_path):
    cache_key = get_cache_key(text)
    with cache_lock:
        tts_cache[cache_key] = audio_path
        if len(tts_cache) > 500:
            for key in list(tts_cache.keys())[:50]:
                del tts_cache[key]

def transcribe_audio_optimized(audio_data, sample_rate):
    if audio_data is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate, format="wav")
            temp_path = temp_file.name
        result = whisper_pipe(temp_path)
        text = result["text"].strip()
        os.unlink(temp_path)
        return text if text else None
    except Exception as e:
        print(f"Error in transcription: {e}")
        return None

def generate_gpt_completion_async(history):
    user_question = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if not user_question:
        return "No user question found."
    if len(history) <= 2:
        cached = get_cached_response(user_question)
        if cached:
            print("Using cached response")
            return cached
    response = ask_umass_assistant(user_question, history)
    if len(history) <= 2:
        cache_response(user_question, response)
    return response

def generate_tts_audio_optimized(text):
    try:
        cached_audio = get_cached_audio(text)
        if cached_audio and os.path.exists(cached_audio):
            print("Using cached TTS audio")
            return cached_audio
        generator = kokoro_pipeline(text, voice='af_heart')
        audio_path = f'/tmp/response_{xxhash.xxh32(text.encode()).hexdigest()}.wav'
        for _, _, audio in generator:
            sf.write(audio_path, audio, 24000)
            break
        cache_audio(text, audio_path)
        return audio_path
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None

def process_audio(audio: tuple, state: AppState):
    return audio, state

def response_optimized(state: AppState, audio: tuple):
    if not audio:
        return state, None
    audio_data, sample_rate = audio[1], audio[0]
    transcription = transcribe_audio_optimized(audio_data, sample_rate)
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."
        state.conversation_history.append({"role": "user", "content": transcription})
        if state.is_first_interaction:
            assistant_message = "Welcome to EDU Bot, your UMass Dartmouth assistant—how can I help you today?"
            state.is_first_interaction = False
        else:
            assistant_message = generate_gpt_completion_async(state.conversation_history)
        state.conversation_history.append({"role": "assistant", "content": assistant_message})
        print(f"User: {transcription}")
        print(f"Assistant: {assistant_message}")
        audio_path = generate_tts_audio_optimized(assistant_message)
        return state, audio_path
    return state, None

def start_recording_user(state: AppState):
    return None

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded");
    var record = document.querySelector('.record-button');
    record.textContent = "🎤 Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) stop.click();
      }
    });
    myvad.start();
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () => {
    document.head.appendChild(script2)
  };
}
"""

js_reset = """
() => {
  var record = document.querySelector('.record-button');
  record.textContent = "🎤 Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

with gr.Blocks(js=js) as demo:
    gr.Markdown("## 🎓 EDU Bot - University of Massachusetts Dartmouth")

    output_audio = gr.Audio(
        label="Voice Response",
        autoplay=True,
        show_label=True,
        container=True
    )

    input_audio = gr.Audio(
        label="🎤 Voice Input",
        sources=["microphone"],
        type="numpy",
        streaming=False,
        show_label=True,
        container=True
    )

    state = gr.State(value=AppState())

    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )

    respond = input_audio.stop_recording(
        response_optimized,
        [state, input_audio],
        [state, output_audio]
    )

    restart = respond.then(
        start_recording_user,
        [state],
        [input_audio]
    ).then(
        lambda state: state,
        state,
        state,
        js=js_reset
    )

if __name__ == "__main__":
    demo.launch(
        server_name="134.88.94.161",
        server_port=7861,
        ssl_certfile="ssl.crt",
        ssl_keyfile="ssl.key",
        ssl_verify=False
    )
