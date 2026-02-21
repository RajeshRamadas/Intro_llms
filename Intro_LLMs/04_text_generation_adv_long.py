import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================
# 1️⃣ Setup Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# 2️⃣ Load Model & Tokenizer
# ==============================
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

# ==============================
# 3️⃣ Define Prompt
# ==============================
prompt = """
Write a detailed and well-structured paragraph explaining how artificial intelligence 
is transforming industries worldwide. Include examples from healthcare, finance, 
manufacturing, and education. Explain benefits, challenges, and future impact.
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# ==============================
# 4️⃣ Generate Long Paragraph
# ==============================
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=350,          # 👈 Controls paragraph length
        temperature=2.0,             # Creativity level
        top_p=0.95,                  # Nucleus sampling
        top_k=50,                    # Limit token choices
        repetition_penalty=1.2,      # Reduce repetition
        no_repeat_ngram_size=3,      # Avoid repeated phrases
        do_sample=True,              # Enable sampling
        pad_token_id=tokenizer.eos_token_id
    )

# ==============================
# 5️⃣ Decode & Print
# ==============================
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n==============================")
print("Generated Long Paragraph:\n")
print(generated_text)
print("\n==============================")