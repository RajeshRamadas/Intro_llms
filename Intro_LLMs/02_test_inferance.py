import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda")
model.to(device)

text = "Artificial intelligence is transforming the world because"
inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True).to(device)

print("Before generation GPU memory (MB):",
      torch.cuda.memory_allocated() / 1024**2)

with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=300,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=True,
        num_return_sequences=3  # Number of outputs you want
    )

print("After generation GPU memory (MB):",
      torch.cuda.memory_allocated() / 1024**2)

for i, sequence in enumerate(output):
    print(f"Generated sequence {i+1}:")
    print(tokenizer.decode(sequence, skip_special_tokens=True))
    print()
