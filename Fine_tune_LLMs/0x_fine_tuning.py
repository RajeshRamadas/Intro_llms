import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

# ----------------------------
# 1️⃣ GPU Check
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total GPU Memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)

# ----------------------------
# 2️⃣ Load SMALLER Dataset (IMPORTANT)
# ----------------------------
train_data = load_dataset("imdb", split="train[:5000]")
test_data = load_dataset("imdb", split="test[:2000]")

print("Train size:", len(train_data))
print("Eval size:", len(test_data))

# ----------------------------
# 3️⃣ Load Lightweight Model
# ----------------------------
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)

# ----------------------------
# 4️⃣ Tokenization (Dynamic Padding)
# ----------------------------
def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=128
    )

train_data = train_data.map(tokenize_function, batched=True)
test_data = test_data.map(tokenize_function, batched=True)

train_data = train_data.remove_columns(["text"])
test_data = test_data.remove_columns(["text"])

train_data.set_format("torch")
test_data.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer)

# ----------------------------
# 5️⃣ Metrics
# ----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

# ----------------------------
# 6️⃣ Training Arguments (Optimized for 4GB GPU)
# ----------------------------
training_args = TrainingArguments(
    output_dir="./results",

    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,

    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,

    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    fp16=True,
    dataloader_num_workers=0,

    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,

    report_to="none"
)

# ----------------------------
# 7️⃣ Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# ----------------------------
# 8️⃣ Train
# ----------------------------
torch.cuda.empty_cache()
trainer.train()

# ----------------------------
# 9️⃣ Evaluate
# ----------------------------
metrics = trainer.evaluate()
print("\nFinal Evaluation Metrics:")
print(metrics)

# ----------------------------
# 🔟 GPU Memory Tracking
# ----------------------------
if torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak GPU Memory Used: {peak_mem:.2f} GB")

# ----------------------------
# 1️⃣1️⃣ Custom Inference
# ----------------------------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "POSITIVE" if pred == 1 else "NEGATIVE"

print("\nCustom Testing:")
print("Test 1:", predict("This movie was absolutely amazing!"))
print("Test 2:", predict("I regret watching this film."))