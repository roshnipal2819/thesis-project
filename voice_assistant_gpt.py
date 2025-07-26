import io
import os
import time
import traceback
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

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ['ESPEAK_DATA_PATH'] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"
kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Using CPU.")

def transcribe_audio(file_name):
    if file_name is None:
        return None
    try:
        from transformers import pipeline
        whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch.float16,
            device=device
        )
        result = whisper_pipe(file_name)
        text = result["text"].strip()
        print("Transcription:", text)
        if len(text) == 0 or text.isspace():
            return None
        return text
    except Exception as e:
        print(f"Error in local Whisper transcription: {e}")
        return f"Error in transcription: {str(e)}"

def generate_gpt_completion(history):
    # Compose the conversation history into a single user question (last user message)
    # Optionally, you could concatenate all user messages, but for now, use the last one
    user_question = None
    for msg in reversed(history):
        if msg["role"] == "user":
            user_question = msg["content"]
            break
    if not user_question:
        return "No user question found."
    return ask_umass_assistant(user_question)

def generate_tts_audio(text):
    """Generate audio using Kokoro TTS"""
    try:
        generator = kokoro_pipeline(text, voice='af_heart')
        audio_path = f'/tmp/response_{xxhash.xxh32(text.encode()).hexdigest()}.wav'
        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(audio_path, audio, 24000)
            break
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

def response(state: AppState, audio: tuple):
    if not audio:
        return state, None

    file_name = f"/tmp/{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav"
    sf.write(file_name, audio[1], audio[0], format="wav")
    
    transcription = transcribe_audio(file_name)
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."

        state.conversation_history.append({"role": "user", "content": transcription})

        assistant_message = generate_gpt_completion(state.conversation_history)
        state.conversation_history.append({"role": "assistant", "content": assistant_message})
        
        print(f"User: {transcription}")
        print(f"Assistant: {assistant_message}")
        
        audio_path = generate_tts_audio(assistant_message)
        
        try:
            os.remove(file_name)
        except:
            pass
            
        return state, audio_path
    
    return state, None

def start_recording_user(state: AppState):
    return None

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c100="#82000019",
        c200="#82000033",
        c300="#8200004c",
        c400="#82000066",
        c50="#8200007f",
        c500="#8200007f",
        c600="#82000099",
        c700="#820000b2",
        c800="#820000cc",
        c900="#820000e5",
        c950="#820000f2",
    ),
    secondary_hue="rose",
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
    gr.Markdown("# 🎤 Voice Assistant (GPT API Model)")
    gr.Markdown("Speak naturally and the assistant will respond with voice!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎵 Assistant Response")
            output_audio = gr.Audio(
                label="Voice Response",
                autoplay=True,
                show_label=True,
                container=True
            )
    
    with gr.Row():
        input_audio = gr.Audio(
            label="🎤 Voice Input",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )
    
    with gr.Row():
        cancel = gr.Button("🔄 New Conversation", variant="stop", size="lg")
        status = gr.Textbox(label="Status", value="Ready to listen...", interactive=False)
    
    state = gr.State(value=AppState())
    
    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
    
    respond = input_audio.stop_recording(
        response, 
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
        lambda: (AppState(), gr.Audio(recording=False), "Ready to listen..."),
        None,
        [state, input_audio, status],
        cancels=[respond, restart],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="134.88.94.161",
        server_port=7861, 
        ssl_certfile="ssl.crt", 
        ssl_keyfile="ssl.key", 
        ssl_verify=False
    ) 