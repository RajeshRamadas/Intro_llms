from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

generator = pipeline(
    "text-generation",
    model="gpt2",
    device=device
)

prompt = "Artificial intelligence is transforming the world because"

output = generator(
    prompt,
    max_new_tokens=100,
    temperature=0.9,
    top_p=0.95,
    do_sample=True
)

print(output[0]["generated_text"])

