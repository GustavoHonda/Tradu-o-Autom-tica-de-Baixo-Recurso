from src.prep import prep_data
from src.model import load_model_and_tokenizer
from src.train import train_model
from src.metrics import compute_metrics
from src.utils import create_dir
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128
NUM_BEAMS = 4

def build_prompts(texts) :
    return [f"resume: {t}" for t in texts]

def generate_batches(model, tokenizer, texts, batch_size):
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        if isinstance(batch, pd.Series):
            batch = batch.tolist()

        prompts = build_prompts(batch)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    return predictions


def save_outputs(x, y, preds, out_name, metrics_name):
    df = pd.DataFrame({
        "Input": list(x),
        "Predicted": preds,
        "Actual": list(y)
    })

    create_dir("./outputs")
    path_out = f"./outputs/{out_name}"
    df.to_csv(path_out, index=False)

    create_dir("./metrics")
    path_metrics = f"./metrics/{metrics_name}"
    compute_metrics(preds, list(y), path_metrics)

    return df


def evaluate_model(model, tokenizer, x_test, y_test, batch_size=8,name="few_shot"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    predictions = []

    for i in tqdm(range(0, len(x_test), batch_size)):
        batch = x_test[i:i + batch_size]
        prompts = [f"resume: {t}" for t in batch]

        inputs = tokenizer(prompts,
                           return_tensors="pt",
                           padding=True,
                           truncation=True,
                           max_length=128).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    # Salva resultados
    df = pd.DataFrame({"Input": x_test, "Predicted": predictions, "Actual": y_test})
    create_dir("./outputs")
    df.to_csv(f"./outputs/{name}_outputs.csv", index=False)

    # Avalia métricas
    create_dir("./metrics")
    compute_metrics(predictions, list(y_test), f"./metrics/{name}_metrics.txt")
    return df

def tokenize_dataset(x_train, y_train, x_val, y_val, tokenizer, max_length=128):
    # Tokenizar entradas
    train_encodings = tokenizer(
        list(x_train),
        text_target=list(y_train),
        padding=True,
        truncation=True,
        max_length=max_length
    )
    val_encodings = tokenizer(
        list(x_val),
        text_target=list(y_val),
        padding=True,
        truncation=True,
        max_length=max_length
    )
    return {"train": train_encodings, "validation": val_encodings}

def few_shot_train(model, tokenizer, x_train, y_train, x_val, y_val, k=16, name="mt5-finetuned"):
    # Seleciona k exemplos
    df_train = pd.DataFrame({"Input": x_train, "Target": y_train}).sample(n=k, random_state=42)
    df_val   = pd.DataFrame({"Input": x_val, "Target": y_val})

    # Tokenizar
    tokenized_dataset = tokenize_dataset(df_train["Input"], df_train["Target"],
                                         df_val["Input"], df_val["Target"], tokenizer)

    # Treinar
    model, tokenizer = train_model(model, tokenizer, tokenized_dataset, name = "mt5-finetuned")
    return model, tokenizer


def few_shot(path, k=16, batch_size=8):
    x_train, x_val, x_test, y_train, y_val, y_test = prep_data(path)

    model, tokenizer = load_model_and_tokenizer()

    # Fine-tuning few-shot
    model, tokenizer = few_shot_train(model, tokenizer, x_train, y_train, x_val, y_val, k=k, name=f"mt5-finetuned")

    # Avaliação no teste
    df_results = evaluate_model(model, tokenizer, x_test, y_test, batch_size=batch_size,name="few_shot")

    return df_results



def few_shot_reverse(path, k=16, batch_size=8):
    x_train, x_val, x_test, y_train, y_val, y_test = prep_data(path)

    model, tokenizer = load_model_and_tokenizer()

    # Fine-tuning few-shot
    
    model, tokenizer = few_shot_train(model, tokenizer, y_train, x_train, y_val, x_val, k=k, name=f"mt5-finetuned-revesed")

    # Avaliação no teste
    df_results = evaluate_model(model, tokenizer, x_test, y_test, batch_size=batch_size,name="few_shot_reverse")

    return df_results

def zero_shot(path: str, batch_size: int = 8):
    _, _, x_test, _, _, y_test = prep_data(path)
    model, tokenizer = load_model_and_tokenizer()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    preds = generate_batches(model, tokenizer, x_test, batch_size)

    return save_outputs(
        x_test, y_test, preds,
        out_name="zero_shot_outputs.csv",
        metrics_name="zero_shot_metrics.txt"
    )

def zero_shot_reverse(path: str, batch_size: int = 8):
    _, _, x_test, _, _, y_test = prep_data(path)
    model, tokenizer = load_model_and_tokenizer()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    preds = generate_batches(model, tokenizer, y_test, batch_size)

    return save_outputs(
        y_test, x_test, preds,
        out_name="zero_shot_reverse_outputs.csv",
        metrics_name="zero_shot_reverse_metrics.txt"
    )