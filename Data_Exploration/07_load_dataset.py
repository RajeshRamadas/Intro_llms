from torchtext.datasets import IMDB

train_iter = IMDB(split="train")

train_list = list(train_iter)

for i in range(5):
    label, text = train_list[i]
    print(f"Sample {i+1}")
    print("Label:", label)
    print("Text:", text[:200])
    print("-" * 50)