from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

text = """
Artificial intelligence is rapidly transforming industries worldwide.
Companies are adopting AI to automate processes, improve efficiency,
and create innovative products. From healthcare to finance,
AI is driving significant technological advancements.
"""

summary = summarizer(
    text,
    max_new_tokens=60,
    min_length=20,
    num_beams=2,
    length_penalty=1.0,
    no_repeat_ngram_size=3,
    early_stopping=True,
    do_sample=False
)

print(summary)
