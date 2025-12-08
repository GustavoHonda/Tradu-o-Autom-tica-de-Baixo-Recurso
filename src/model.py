from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

cache_dir = "D:/huggingface_cache"

def load_model_and_tokenizer():
    model_name = "google/mt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name,  cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
    return model, tokenizer