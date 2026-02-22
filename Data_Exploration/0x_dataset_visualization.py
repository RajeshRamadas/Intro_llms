from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Unbelievably good movie!"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)


print(tokenizer.cls_token)
print(tokenizer.sep_token)
print(tokenizer.pad_token)

decoded = tokenizer.decode(token_ids)
print(decoded)
