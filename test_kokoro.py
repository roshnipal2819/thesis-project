# from kokoro import KPipeline
# import soundfile as sf
# import torch
# import gradio as gr
# import os

# os.environ['ESPEAK_DATA_PATH'] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"

# print("Device:", torch.cuda.get_device_name(0))

# pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

# def generate_and_play(user_text):
#     if not user_text.strip():
#         return None, "Please enter some text to generate speech."
    
#     try:
#         generator = pipeline(user_text, voice='af_heart')
#         audio_path = 'response.wav'
#         for i, (gs, ps, audio) in enumerate(generator):
#             sf.write(audio_path, audio, 24000)
#             break
#         return audio_path, f"Generated audio for: '{user_text}'"
#     except Exception as e:
#         return None, f"Error generating audio: {str(e)}"

# iface = gr.Interface(
#     fn=generate_and_play,
#     inputs=gr.Textbox(
#         label="Enter your message",
#         placeholder="Type your message here...",
#         lines=3
#     ),
#     outputs=[
#         gr.Audio(label="Voice Assistant Response", autoplay=True),
#         gr.Textbox(label="Status", interactive=False)
#     ],
#     title="Kokoro Voice Assistant",
#     description="Enter your message and the assistant will respond with voice automatically.",
#     examples=[
#         ["Hello, how are you today?"],
#         ["What is the weather like?"],
#         ["Tell me a joke."],
#         ["Kokoro is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient."]
#     ]
# )

# if __name__ == "__main__":
#     iface.launch(server_name="134.88.94.161", server_port=7861, ssl_certfile="ssl.crt", ssl_keyfile="ssl.key", ssl_verify=False)


from kokoro import KPipeline
import soundfile as sf
import torch
import os
import time
import pynvml

# Set ESpeak path
os.environ['ESPEAK_DATA_PATH'] = "/usr/lib/x86_64-linux-gnu/espeak-ng-data"

# GPU name
print("Device:", torch.cuda.get_device_name(0))

# GPU profiling helper
def gpu_stats():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_util_percent": util_rate.gpu,
        "vram_used_gb": round(mem_info.used / (1024 ** 3), 2)
    }

# Load Kokoro model (no warm-up, no optimization)
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

# Text for inference
user_text = "going along slushy country roads and speaking to damp audiences in draughty schoolrooms day after day for a fortnight he'll have to put in an appearance at some place of worship on sunday morning and he can come to us immediately afterwards."

# Run inference with profiling
try:
    start_time = time.time()

    generator = pipeline(user_text, voice='af_heart')
    audio_path = 'response.wav'

    for i, (gs, ps, audio) in enumerate(generator):
        sf.write(audio_path, audio, 24000)
        break

    end_time = time.time()
    latency_ms = round((end_time - start_time) * 1000, 1)
    gpu_info = gpu_stats()

    print(f"Text: {user_text}")
    print(f"Output saved to: {audio_path}")
    print(f"Latency: {latency_ms} ms")
    print(f"GPU Utilization: {gpu_info['gpu_util_percent']}%")
    print(f"VRAM Used: {gpu_info['vram_used_gb']} GB")

except Exception as e:
    print(f"Error generating audio: {str(e)}")
