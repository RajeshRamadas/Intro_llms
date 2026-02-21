import torch
from transformers import AutoConfig, AutoTokenizer, AutoModel

def inspect_model(model_name):
    print("\n" + "="*60)
    print(f"Inspecting Model: {model_name}")
    print("="*60)

    # Load config only (lightweight)
    config = AutoConfig.from_pretrained(model_name)

    print("\n🔹 Model Type:", config.model_type)
    print("🔹 Is Encoder-Decoder:", config.is_encoder_decoder)
    print("🔹 Hidden Size:", getattr(config, "hidden_size", "N/A"))
    print("🔹 Num Layers:", getattr(config, "num_hidden_layers", "N/A"))
    print("🔹 Num Attention Heads:", getattr(config, "num_attention_heads", "N/A"))
    print("🔹 Vocab Size:", getattr(config, "vocab_size", "N/A"))
    print("🔹 Max Position Embeddings:", getattr(config, "max_position_embeddings", "N/A"))

    print("\n🔹 Full Config:")
    print(config)

def model_category(model_name):
    config = AutoConfig.from_pretrained(model_name)

    if config.is_encoder_decoder:
        print(f"{model_name} → Encoder-Decoder (Seq2Seq)")
    elif config.model_type in ["gpt2", "llama", "falcon"]:
        print(f"{model_name} → Decoder-only (Causal LM)")
    else:
        print(f"{model_name} → Encoder-only (BERT-style)")

def count_parameters(model_name):
    model = AutoModel.from_pretrained(model_name)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name} → {total_params/1e6:.2f} Million parameters")

# 🔥 Inspect Different Models
inspect_model("sshleifer/distilbart-cnn-12-6")
inspect_model("gpt2")
inspect_model("Helsinki-NLP/opus-mt-en-de")
inspect_model("bert-base-uncased")


model_category("gpt2")
model_category("bert-base-uncased")
model_category("sshleifer/distilbart-cnn-12-6")
model_category("Helsinki-NLP/opus-mt-en-de")

count_parameters("gpt2")
count_parameters("sshleifer/distilbart-cnn-12-6")
count_parameters("Helsinki-NLP/opus-mt-en-de")
count_parameters("bert-base-uncased")
