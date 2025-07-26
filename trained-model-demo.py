# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import time

# torch.manual_seed(42)
# if torch.cuda.is_available():
#     print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is NOT available. Using CPU.")

# start_time = time.time()

# model_path = "llama-3.2-3B-Instruct-bnb-4bit-finetuned"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto"
# )

# prompt = "User: I am an international student, and want to know more about CPT/OPT? \nAssistant:"
# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=150,
#     # temperature=0.6,
#     do_sample=True,
#     use_cache=True,
# )
# end_time = time.time()
# latency = end_time - start_time

# response = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(response)
# print(f"\n🔁 Inference Latency: {latency:.2f} seconds ({latency*1000:.0f} ms)")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import pynvml

torch.manual_seed(42)

# Print GPU info
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available. Using CPU.")

# GPU stats function
def gpu_stats():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
    return {
        "gpu_util_percent": util_rate.gpu,
        "vram_used_gb": round(mem_info.used / (1024 ** 3), 2)
    }

# Load model/tokenizer
model_path = "llama-3.2-3B-Instruct-bnb-4bit-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)
model.eval() 
print("Model loaded and set to eval mode.")

prompt = "User: Who is my academic advisor?\nAssistant:"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Start inference timer only (excluding load time)
start_time = time.time()

with torch.inference_mode():  
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,     # reduce token count if not needed
        do_sample=False,        # disable sampling for speed
        use_cache=True,         # already fast, keep it
    )

end_time = time.time()
latency = end_time - start_time
gpu_info = gpu_stats()

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

print(f"\nInference Latency: {latency:.2f} seconds ({latency*1000:.0f} ms)")
print(f"GPU Utilization: {gpu_info['gpu_util_percent']}%")
print(f"VRAM Used: {gpu_info['vram_used_gb']} GB")
