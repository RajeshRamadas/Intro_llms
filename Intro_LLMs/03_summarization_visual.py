import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

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

configs = {
    "Greedy": dict(max_new_tokens=50, do_sample=False),
    "Beam (2)": dict(max_new_tokens=50, num_beams=2, do_sample=False),
    "Sampling T=0.7": dict(max_new_tokens=50, do_sample=True, temperature=0.7),
    "Sampling T=2.0": dict(max_new_tokens=50, do_sample=True, temperature=2.0),
}

for name, params in configs.items():
    print(f"\n=== {name} ===")
    summary = summarizer(text, **params)
    print(summary[0]["summary_text"])
