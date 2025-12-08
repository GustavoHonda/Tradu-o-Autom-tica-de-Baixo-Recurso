from transformers import Trainer, TrainingArguments
import os
os.environ["TORCH_HOME"] = "D:/torch_cache"

import torch
from torch.utils.data import Dataset
from src.utils import create_dir

# Dataset PyTorch para seq2seq
class Seq2SeqDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

def train_model(model, tokenizer, tokenized_dataset):
    """
    Treina um modelo seq2seq (ex.: T5/BART) usando Hugging Face Trainer,
    sem evaluation_strategy e compatível com versões antigas.
    """

    cache_dir = "D:/huggingface_cache"

    # Transformar tokenizações em Dataset PyTorch
    train_dataset = Seq2SeqDataset(tokenized_dataset["train"])
    val_dataset = Seq2SeqDataset(tokenized_dataset["validation"]) if "validation" in tokenized_dataset else None

    # Detecta se há GPU
    fp16_flag = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir="D:/model/mt5-finetuned",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        fp16=fp16_flag,
        save_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    trainer.train()

    # Salva modelo e tokenizer
    create_dir("./model")
    trainer.save_model("./model/mt5-finetuned")
    tokenizer.save_pretrained("./model/mt5-finetuned")

    print("Treinamento concluído e modelo salvo em ./model/mt5-finetuned")
    return model, tokenizer
