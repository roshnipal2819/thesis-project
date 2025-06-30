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
import spaces
import xxhash
from datasets import Audio
import torch
from transformers import pipeline
import openai

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device=device)

def transcribe_audio(file_name):
    if file_name is None:
        return None
    try:
        result = whisper_pipe(file_name)
        text = result["text"].strip()
        print("Transcription:", text)
        if len(text) == 0 or text.isspace():
            return None
        return text
    except Exception as e:
        print(f"Error in local Whisper transcription: {e}")
        return f"Error in transcription: {str(e)}"

def generate_chat_completion(history):
    messages = [
        {
            "role": "system",
            "content": "In conversation with the user, ask questions to estimate and provide (1) total calories, (2) protein, carbs, and fat in grams, (3) fiber and sugar content. Only ask *one question at a time*. Be conversational and natural.",
        }
    ] + history
    try:
        client = openai.OpenAI()
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error in generating chat completion: {str(e)}"

@dataclass
class AppState:
    conversation: list = field(default_factory=list)
    stopped: bool = False
    model_outs: any = None

def process_audio(audio: tuple, state: AppState):
    return audio, state

@spaces.GPU(duration=40, progress=gr.Progress(track_tqdm=True))
def response(state: AppState, audio: tuple):
    if not audio:
        return AppState()

    file_name = f"/tmp/{xxhash.xxh32(bytes(audio[1])).hexdigest()}.wav"

    sf.write(file_name, audio[1], audio[0], format="wav")
    # Transcribe the audio file
    transcription = transcribe_audio(file_name)
    if transcription:
        if transcription.startswith("Error"):
            transcription = "Error in audio transcription."

        # Append the user's message in the proper format
        state.conversation.append({"role": "user", "content": transcription})

        # Generate assistant response
        assistant_message = generate_chat_completion(state.conversation)

        # Append the assistant's message in the proper format
        state.conversation.append({"role": "assistant", "content": assistant_message})

        print(state.conversation)
        os.remove(file_name)
    return state, state.conversation

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
    record.textContent = "Just Start Talking!"
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
  record.textContent = "Just Start Talking!"
  record.style = "width: fit-content; padding-right: 0.5vw;"
}
"""

with gr.Blocks(theme=theme, js=js) as demo:
    with gr.Row():
        input_audio = gr.Audio(
            label="Input Audio",
            sources=["microphone"],
            type="numpy",
            streaming=False,
            waveform_options=gr.WaveformOptions(waveform_color="#B83A4B"),
        )
    with gr.Row():
        chatbot = gr.Chatbot(label="Conversation", type="messages")
    state = gr.State(value=AppState())
    stream = input_audio.start_recording(
        process_audio,
        [input_audio, state],
        [input_audio, state],
    )
    respond = input_audio.stop_recording(
        response, [state, input_audio], [state, chatbot]
    )
    restart = respond.then(start_recording_user, [state], [input_audio]).then(
        lambda state: state, state, state, js=js_reset
    )
    cancel = gr.Button("New Conversation", variant="stop")
    cancel.click(
        lambda: (AppState(), gr.Audio(recording=False)),
        None,
        [state, input_audio],
        cancels=[respond, restart],
    )

if __name__ == "__main__":
    demo.launch(server_name="134.88.94.161",
    server_port=7063,
    ssl_certfile="ssl.crt",
    ssl_keyfile="ssl.key",
    ssl_verify=False)