from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

print(type(summarizer))
print(summarizer.task)
print(summarizer.model.__class__)