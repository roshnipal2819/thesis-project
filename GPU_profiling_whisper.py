# import torch
# from transformers import pipeline
# import time
# import pynvml

# print("Device:", torch.cuda.get_device_name(0))

# def gpu_stats():
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
#     return {
#         "gpu_util_percent": util_rate.gpu,
#         "vram_used_gb": round(mem_info.used / (1024 ** 3), 2)
#     }

# device = 0 if torch.cuda.is_available() else -1

# pipe = pipeline(
#     "automatic-speech-recognition",
#     model="openai/whisper-large-v3-turbo",
#     torch_dtype=torch.float16,
#     device=device
# )

# audio_path="/home/roshni/Project/Whisper/audio_files/sample1.flac"

# start_time = time.time()
# result = pipe(audio_path)
# end_time = time.time()

# latency_ms = round((end_time - start_time) * 1000, 1)
# gpu_info = gpu_stats()

# print("Transcription:", result["text"])
# print(f"Latency (ms): {latency_ms}")
# print(f"GPU Utilization (%): {gpu_info['gpu_util_percent']}")
# print(f"VRAM Used (GB): {gpu_info['vram_used_gb']}")


import os
import json
import time
import pynvml
from dotenv import load_dotenv
from openai import OpenAI

# Load OpenAI API Key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Initialize GPU profiler
try:
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "No GPU Available"
except ImportError:
    device_name = "torch not installed"

def gpu_stats():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_util_percent": util_rate.gpu,
            "vram_used_gb": round(mem_info.used / (1024 ** 3), 2)
        }
    except Exception as e:
        return {
            "gpu_util_percent": -1,
            "vram_used_gb": -1,
            "error": str(e)
        }

# Load few-shot examples
with open("data.json", "r") as f:
    data = json.load(f)

# Query OpenAI Assistant
def ask_umass_assistant(user_question):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful voice assistant for the University of Massachusetts Dartmouth, College of Engineering. Respond clearly using official university policies."
            }
        ]
        for pair in data[:50]:
            messages.append({"role": "user", "content": pair["conversation"][0]["content"]})
            messages.append({"role": "assistant", "content": pair["conversation"][1]["content"]})

        messages.append({"role": "user", "content": user_question})

        start_time = time.time()

        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=300
        )

        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 1)
        gpu_info = gpu_stats()

        print("\n--- Performance Metrics ---")
        print(f"Device: {device_name}")
        print(f"Latency: {latency_ms} ms")
        if "error" in gpu_info:
            print(f"GPU Stats Error: {gpu_info['error']}")
        else:
            print(f"GPU Utilization: {gpu_info['gpu_util_percent']}%")
            print(f"VRAM Used: {gpu_info['vram_used_gb']} GB")
        print("---------------------------")

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

# CLI Loop
if __name__ == "__main__":
    print(f"Running on: {device_name}")
    while True:
        try:
            question = input("Ask your question (or type 'exit'): ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            answer = ask_umass_assistant(question)
            print(f"\n🗣️ Assistant: {answer}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
