from transformers import AutoTokenizer

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")

text = "Unbelievably good movie!"

print("BERT:", bert_tok.tokenize(text))
print("GPT2:", gpt2_tok.tokenize(text))
