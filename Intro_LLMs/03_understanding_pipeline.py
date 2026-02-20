from transformers import pipeline
import torch

# Use device as string for compatibility
device = "cuda:0" if torch.cuda.is_available() else "cpu"

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=device
)

text = """
Walking amid Gion's Machiya wooden houses is a mesmerizing experience.
The beautifully preserved structures exuded an old-world charm that transports visitors
back in time, making them feel like they had stepped into a living museum.
The glow of lanterns lining the narrow streets add to the enchanting ambiance,
making each stroll a memorable journey through Japan's rich cultural history.
"""

summary = summarizer(
    text,
    max_length=50,
    min_length=20,
    num_beams=2
)

print(summary[0]["summary_text"])