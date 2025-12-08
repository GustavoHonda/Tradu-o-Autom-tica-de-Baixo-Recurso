from src.prep import prep_data
from src.model import load_model_and_tokenizer
from src.train import train_model
from src.metrics import compute_metrics
from src.utils import create_dir
import os
os.environ["TORCH_HOME"] = "D:/torch_cache"
import torch
import pandas as pd
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


def zero_shot(path: str, batch_size=8):
    _, _, x_test, _, _, y_test = prep_data(path)

    model, tokenizer = load_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = []

    for i in tqdm(range(0, len(x_test), batch_size)):
        batch = x_test.iloc[i:i + batch_size].tolist()
        prompts = [f"resume: {t}" for t in batch]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)

    df = pd.DataFrame({
        "Input": x_test,
        "Predicted": predictions,  
        "Actual": y_test
    })
    create_dir("./outputs")
    df.to_csv("./outputs/zero_shot_outputs.csv", index=False)

    create_dir("./metrics")
    compute_metrics(
        predictions,
        y_test.tolist(),
        "./metrics/zero_shot_metrics.txt"
    )

    return df


def evaluate_model(model, tokenizer, x_test, y_test, batch_size=8):
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
    df.to_csv("./outputs/few_shot_outputs.csv", index=False)

    # Avalia métricas
    create_dir("./metrics")
    compute_metrics(predictions, list(y_test), "./metrics/few_shot_metrics.txt")
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



def few_shot_train(model, tokenizer, x_train, y_train, x_val, y_val, k=16):
    # Seleciona k exemplos
    df_train = pd.DataFrame({"Input": x_train, "Target": y_train}).sample(n=k, random_state=42)
    df_val   = pd.DataFrame({"Input": x_val, "Target": y_val})

    # Tokenizar
    tokenized_dataset = tokenize_dataset(df_train["Input"], df_train["Target"],
                                         df_val["Input"], df_val["Target"], tokenizer)

    # Treinar
    model, tokenizer = train_model(model, tokenizer, tokenized_dataset)
    return model, tokenizer


def few_shot(path, k=16, batch_size=8):
    x_train, x_val, x_test, y_train, y_val, y_test = prep_data(path)

    model, tokenizer = load_model_and_tokenizer()

    # Fine-tuning few-shot
    model, tokenizer = few_shot_train(model, tokenizer, x_train, y_train, x_val, y_val, k=k)

    # Avaliação no teste
    df_results = evaluate_model(model, tokenizer, x_test, y_test, batch_size=batch_size)

    return df_results
