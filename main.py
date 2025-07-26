import io
import os
import time
import traceback
import asyncio
import hashlib
import pickle
from dataclasses import dataclass, field
from dotenv import load_dotenv
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import xxhash
from datasets import Audio
import torch
from kokoro import KPipeline
from API_call import ask_umass_assistant
from transformers import pipeline
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global pipeline initialization (done once at startup)
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

# Response caching
response_cache = {}
tts_cache = {}
cache_lock = threading.Lock()

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Using CPU.")

def get_cache_key(text, cache_type="response"):
    """Generate cache key for text"""
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_response(text):
    """Get cached response if available"""
    cache_key = get_cache_key(text, "response")
    with cache_lock:
        return response_cache.get(cache_key)

def cache_response(text, response):
    """Cache response for future use"""
    cache_key = get_cache_key(text, "response")
    with cache_lock:
        response_cache[cache_key] = response
        # Limit cache size
        if len(response_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(response_cache.keys())[:100]
            for key in oldest_keys:
                del response_cache[key]

def get_cached_audio(text):
    """Get cached TTS audio if available"""
    cache_key = get_cache_key(text, "tts")
    with cache_lock:
        return tts_cache.get(cache_key)

def cache_audio(text, audio_path):
    """Cache TTS audio for future use"""
    cache_key = get_cache_key(text, "tts")
    with cache_lock:
        tts_cache[cache_key] = audio_path
        # Limit cache size
        if len(tts_cache) > 500:
            # Remove oldest entries
            oldest_keys = list(tts_cache.keys())[:50]
            for key in oldest_keys:
                del tts_cache[key]

def transcribe_audio_optimized(audio_data, sample_rate):
    """Optimized transcription using pre-initialized pipeline"""
    if audio_data is None:
        return None
    
    try:
        # Create temporary file in memory if possible
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, sample_rate, format="wav")
            temp_path = temp_file.name
        
        # Use pre-initialized pipeline
        result = whisper_pipe(temp_path)
        text = result["text"].strip()
        
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        print("Transcription:", text)
        if len(text) == 0 or text.isspace():
            return None
        return text
    except Exception as e:
        print(f"Error in transcription: {e}")
        return f"Error in transcription: {str(e)}"

def generate_gpt_completion_async(history):
    """Async GPT completion with caching"""
    # Get the last user message
    user_question = None
    for msg in reversed(history):
        if msg["role"] == "user":
            user_question = msg["content"]
            break
    
    if not user_question:
        return "No user question found."
    
    # Check cache first
    cached_response = get_cached_response(user_question)
    if cached_response:
        print("Using cached response")
        return cached_response
    
    # Generate new response
    response = ask_umass_assistant(user_question)
    
    # Cache the response
    cache_response(user_question, response)
    
    return response

def generate_tts_audio_optimized(text):
    """Optimized TTS with caching"""
    try:
        # Check cache first
        cached_audio = get_cached_audio(text)
        if cached_audio and os.path.exists(cached_audio):
            print("Using cached TTS audio")
            return cached_audio
        
        # Generate new audio
        generator = kokoro_pipeline(text, voice='af_heart')
        audio_path = f'/tmp/response_{xxhash.xxh32(text.encode()).hexdigest()}.wav'
        
        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(audio_path, audio, 24000)
            break
        
        # Cache the audio
        cache_audio(text, audio_path)
        
        return audio_path
    except Exception as e:
        print(f"Error in TTS generation: {e}")
        return None

@dataclass
class AppState:
    conversation: list = field(default_factory=list)  
    conversation_history: list = field(default_factory=list) 
    stopped: bool = False
    model_outs: any = None

def process_audio(audio: tuple, state: AppState):
    return audio, state

def response_optimized(state: AppState, audio: tuple):
    """Optimized response function with reduced latency"""
    if not audio:
        return state, None

    start_time = time.time()
    
    # Extract audio data directly without file I/O
    audio_data, sample_rate = audio[1], audio[0]
    
    # Step 1: Transcribe (optimized)
    transcription_start = time.time()
    transcription = transcribe_audio_optimized(audio_data, sample_rate)
    transcription_time = time.time() - transcription_start
    
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."

        state.conversation_history.append({"role": "user", "content": transcription})

        # Step 2: Generate response (with caching)
        response_start = time.time()
        assistant_message = generate_gpt_completion_async(state.conversation_history)
        response_time = time.time() - response_start
        
        state.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        print(f"User: {transcription}")
        print(f"Assistant: {assistant_message}")
        
        # Step 3: Generate TTS (with caching)
        tts_start = time.time()
        audio_path = generate_tts_audio_optimized(assistant_message)
        tts_time = time.time() - tts_start
        
        total_time = time.time() - start_time
        
        print(f"⏱️ Latency Breakdown:")
        print(f"  Transcription: {transcription_time*1000:.0f}ms")
        print(f"  Response Generation: {response_time*1000:.0f}ms")
        print(f"  TTS Generation: {tts_time*1000:.0f}ms")
        print(f"  Total Latency: {total_time*1000:.0f}ms")
            
        return state, audio_path
    
    return state, None

def start_recording_user(state: AppState):
    return None

# Preload common responses for faster access
def preload_common_responses():
    """Preload common responses to reduce API calls"""
    common_questions = [
        "Hello",
        "How are you?",
        "What is your name?",
        "Thank you",
        "Goodbye"
    ]
    
    print("Preloading common responses...")
    for question in common_questions:
        response = ask_umass_assistant(question)
        cache_response(question, response)
    print("Common responses preloaded!")

# Initialize preloading in background
threading.Thread(target=preload_common_responses, daemon=True).start()

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#1e3a8a19",
        c200="#1e3a8a33",
        c300="#1e3a8a4c",
        c400="#1e3a8a66",
        c50="#1e3a8a7f",
        c500="#1e3a8a7f",
        c600="#1e3a8a99",
        c700="#1e3a8ab2",
        c800="#1e3a8acc",
        c900="#1e3a8ae5",
        c950="#1e3a8af2",
    ),
    secondary_hue="blue",
    neutral_hue="stone",
)

js = """
async function main() {
  const script1 = document.createElement("script");
  script1.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.js";
  document.head.appendChild(script1)
  const script2 = document.createElement("script");
  script2.onload = async () =>  {
    console.log("vad loaded") ;
    var record = document.querySelector('.record-button');
    record.textContent = "🎤 Start Talking!"
    record.style = "width: fit-content; padding-right: 0.5vw;"
    const myvad = await vad.MicVAD.new({
      onSpeechStart: () => {
        var record = document.querySelector('.record-button');
        var player = document.querySelector('#streaming-out')
        if (record != null && (player == null || player.paused)) {
          console.log(record);
          record.click();
        }
      },
      onSpeechEnd: (audio) => {
        var stop = document.querySelector('.stop-button');
        if (stop != null) {
          console.log(stop);
          stop.click();
        }
      }
    })
    myvad.start()
  }
  script2.src = "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.7/dist/bundle.min.js";
  script1.onload = () =>  {
    console.log("onnx loaded") 
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

with gr.Blocks(theme=theme, js=js) as demo:
    # Header with UMass branding
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
                        border-radius: 15px; margin-bottom: 25px; box-shadow: 0 6px 20px rgba(0,0,0,0.15);">
                <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 15px;">
                    <div style="font-size: 48px; margin-right: 15px;">🎓</div>
                    <div style="font-size: 48px;">🤖</div>
                </div>
                <h1 style="color: white; margin: 0; font-size: 2.8em; font-weight: bold; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    EDUBOT
                </h1>
                <p style="color: #e0e7ff; margin: 12px 0 0 0; font-size: 1.3em; font-weight: 600;">
                    University of Massachusetts Dartmouth
                </p>
                <p style="color: #c7d2fe; margin: 8px 0 0 0; font-size: 1.1em;">
                    Intelligent Voice Assistant for University Students
                </p>
            </div>
            """)
    
    # Main content area with UMass styling
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.08) 0%, rgba(59, 130, 246, 0.05) 100%); 
                        padding: 20px; border-radius: 12px; border-left: 5px solid #1e3a8a; 
                        margin-bottom: 20px; box-shadow: 0 2px 10px rgba(30, 58, 138, 0.1);">
                <h3 style="color: #1e3a8a; margin: 0 0 15px 0; display: flex; align-items: center; font-size: 1.4em;">
                    <span style="font-size: 28px; margin-right: 12px;">🎵</span>
                    Assistant Response
                </h3>
                <p style="color: #6b7280; margin: 0; font-size: 0.9em;">UMass Dartmouth AI will respond here</p>
            </div>
            """)
            output_audio = gr.Audio(
                label="Voice Response",
                autoplay=True,
                show_label=False,
                container=True
            )
    
    with gr.Row():
        gr.HTML("""
        <div style="background: linear-gradient(135deg, rgba(30, 58, 138, 0.08) 0%, rgba(59, 130, 246, 0.05) 100%); 
                    padding: 20px; border-radius: 12px; border-left: 5px solid #1e3a8a; 
                    margin-bottom: 20px; width: 100%; box-shadow: 0 2px 10px rgba(30, 58, 138, 0.1);">
            <h3 style="color: #1e3a8a; margin: 0 0 15px 0; display: flex; align-items: center; 
                        justify-content: space-between; font-size: 1.4em;">
                <span style="display: flex; align-items: center;">
                    <span style="font-size: 28px; margin-right: 12px;">🎤</span>
                    Voice Input
                </span>
                <span style="font-size: 16px; color: #6b7280; background: rgba(30, 58, 138, 0.1); 
                            padding: 4px 12px; border-radius: 15px;">UMass Dartmouth</span>
            </h3>
            <p style="color: #6b7280; margin: 0; font-size: 0.9em;">Speak your questions naturally</p>
        </div>
        """)
        input_audio = gr.Audio(
            label="🎤 Voice Input",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#1e3a8a"),
            show_label=False,
            container=True
        )
    
    with gr.Row():
        with gr.Column(scale=1):
            cancel = gr.Button(
                "🔄 New Conversation", 
                variant="primary", 
                size="lg",
                elem_classes=["umass-button"]
            )
        with gr.Column(scale=1):
            status = gr.Textbox(
                label="System Status", 
                value="🎓 EDUBOT Ready - UMass Dartmouth Assistant Online", 
                interactive=False,
                elem_classes=["umass-status"]
            )
    
    # Add custom CSS for UMass styling
    gr.HTML("""
    <style>
    .umass-button {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%) !important;
        border: none !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3) !important;
        transition: all 0.3s ease !important;
        font-size: 1.1em !important;
        padding: 12px 24px !important;
    }
    .umass-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4) !important;
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%) !important;
    }
    .umass-status {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        border: 2px solid #1e3a8a !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        color: #1e3a8a !important;
        font-size: 1.1em !important;
    }
    .umass-status label {
        color: #1e3a8a !important;
        font-weight: bold !important;
        font-size: 1.1em !important;
    }
    /* Hide default Gradio footer */
    footer {
        display: none !important;
    }
    .gradio-container footer {
        display: none !important;
    }
    /* Custom footer positioning */
    .custom-footer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1000 !important;
        background: transparent !important;
        padding: 10px 20px !important;
        text-align: center !important;
    }
    /* Add bottom margin to main content to prevent overlap */
    .main-container {
        margin-bottom: 50px !important;
    }
    </style>
    """)

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
    
    cancel.click(
        lambda: (AppState(), None, "🎓 EDUBOT Ready - UMass Dartmouth Assistant Online"),
        None,
        [state, input_audio, status],
        cancels=[respond, restart],
    )

    # Custom footer at the very bottom
    gr.HTML("""
    <div class="custom-footer">
        <p style="color: #6b7280; margin: 0; font-size: 0.9em; font-weight: 400;">
            Design and Developed by Roshni
        </p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="134.88.94.161",
        server_port=7861, 
        ssl_certfile="ssl.crt", 
        ssl_keyfile="ssl.key", 
        ssl_verify=False
    ) 