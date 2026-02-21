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

def test_config(name, **params):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\n=== {name} ===")
    
    summarizer(text, **params)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Peak GPU memory (MB): {peak_mem:.2f}")

configs = {
    "Greedy": dict(max_new_tokens=50, do_sample=False),
    "Beam (2)": dict(max_new_tokens=50, num_beams=2, do_sample=False),
    "Beam (4)": dict(max_new_tokens=50, num_beams=4, do_sample=False),
    "Sampling": dict(max_new_tokens=50, do_sample=True, temperature=0.9),
    "Greedy (200)": dict(max_new_tokens=200, do_sample=False),
    "Beam (2) (200)": dict(max_new_tokens=200, num_beams=2, do_sample=False),
    "Beam (4) (200)": dict(max_new_tokens=200, num_beams=4, do_sample=False),
    "Sampling (200)": dict(max_new_tokens=200, do_sample=True, temperature=0.9),
}
print("============================================")
for name, params in configs.items():
    print("----------------------------------------")
    test_config(name, **params)
print("============================================")
