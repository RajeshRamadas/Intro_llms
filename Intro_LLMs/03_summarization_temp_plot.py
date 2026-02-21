# 📊 Plot: Memory vs Temperature (Sampling Enabled)

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

text = """
Artificial intelligence is rapidly transforming industries worldwide.
Companies are adopting AI to automate processes, improve efficiency,
and create innovative products across healthcare, finance, and manufacturing.
"""

inputs = tokenizer(text, return_tensors="pt").to(device)

temperature_values = [0.5, 0.7, 1.0, 1.5, 2.0]
memory_usage = []

for temp in temperature_values:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=temp
        )

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    memory_usage.append(peak_mem)

    print(f"temperature={temp} → Peak GPU Memory: {peak_mem:.2f} MB")

plt.figure()
plt.plot(temperature_values, memory_usage)
plt.xlabel("Temperature")
plt.ylabel("Peak GPU Memory (MB)")
plt.title("Memory Usage vs Temperature")
plt.show()
