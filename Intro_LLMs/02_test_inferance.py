import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda")
model.to(device)

text = "Artificial intelligence is transforming the world because"
inputs = tokenizer(text, return_tensors="pt").to(device)

print("Before generation GPU memory (MB):",
      torch.cuda.memory_allocated() / 1024**2)

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_length=100
    )

print("After generation GPU memory (MB):",
      torch.cuda.memory_allocated() / 1024**2)

print(tokenizer.decode(output[0], skip_special_tokens=True))