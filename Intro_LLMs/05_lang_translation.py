import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# 1️⃣ Setup Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 2️⃣ Load Translation Model
# ==============================
model_name = "Helsinki-NLP/opus-mt-en-de"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to(device)
model.eval()

# ==============================
# 3️⃣ Input Text
# ==============================
text = """
No single organization or sector can solve the challenges behind large-scale migration alone.
This makes it imperative to operate from a multi-sectorial approach with partners who listen, learn and work closely with one another. Each contributes specific skills and resources: funding, consulting, expert advice, networks, space, and brilliant thinking.
The result is an ecosystem that enables scaling the best solutions for migration throughout the globe.
"""

inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)

# ==============================
# 4️⃣ Generate Translation
# ==============================
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        num_beams=4,                # Beam search for accuracy
        length_penalty=1.0,
        early_stopping=True
    )

translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# 5️⃣ Print Output
# ==============================
print("\n==============================")
print("Translated Text (German):\n")
print(translated_text)
print("==============================")