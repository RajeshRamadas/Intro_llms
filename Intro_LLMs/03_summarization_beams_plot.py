# 📊 Plot: Memory vs num_beams

import torch
import matplotlib.pyplot as plt
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
print("Using device:", "GPU" if device == 0 else "CPU")

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

text = """
Artificial intelligence is rapidly transforming industries worldwide.
Companies are adopting AI to automate processes, improve efficiency,
and create innovative products across healthcare, finance, and manufacturing.
"""

beam_values = [1, 2, 3, 4]
memory_usage = []

for beams in beam_values:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    summarizer(
        text,
        max_new_tokens=50,
        num_beams=beams,
        do_sample=False
    )

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    memory_usage.append(peak_mem)

    print(f"num_beams={beams} → Peak GPU Memory: {peak_mem:.2f} MB")

plt.figure()
plt.plot(beam_values, memory_usage)
plt.xlabel("num_beams")
plt.ylabel("Peak GPU Memory (MB)")
plt.title("Memory Usage vs num_beams")
plt.show()