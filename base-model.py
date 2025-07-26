import os
import time
import torch
import logging
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# --- Suppress verbose logging ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("unsloth").setLevel(logging.CRITICAL)

# --- Optional: Extract only assistant reply ---
def extract_assistant_response(text: str) -> str:
    if "assistant" in text.lower():
        return text.split("assistant", 1)[-1].strip().lstrip(":").strip()
    return text.strip()

# --- Load base model (no fine-tuning) ---
model_path = "unsloth/llama-3.2-3B-Instruct-bnb-4bit"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True
)

# Apply chat template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")
tokenizer.pad_token = tokenizer.eos_token
FastLanguageModel.for_inference(model)

# Optional: compile for better performance
model = torch.compile(model)

# --- Prompt ---
messages = [
    {"role": "user", "content": "What is my campus address for receiving mail?, in 2-3 lines."}
]

# --- Tokenize ---
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True
).to(model.device)

# --- Generate with latency measurement ---
start_time = time.time()
outputs = model.generate(
    input_ids=inputs,
    attention_mask=inputs != tokenizer.pad_token_id,
    max_new_tokens=256,
    use_cache=True,
    temperature=0.6,
    min_p=0.1,
)
end_time = time.time()

# --- Decode + show ---
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = extract_assistant_response(text)

# --- Print result and latency ---
print("\n=== Assistant Response ===\n")
print(response)
print(f"\n⏱️ Inference Latency: {end_time - start_time:.2f} sec ({(end_time - start_time)*1000:.0f} ms)")
