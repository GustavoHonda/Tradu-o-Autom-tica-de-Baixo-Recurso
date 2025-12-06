from src.prep import prep_data
from src.model import load_model_and_tokenizer
from src.train import train_model
from src.metrics import compute_metrics
from src.utils import create_dir
import torch
import pandas as pd
from tqdm import tqdm


def zero_shot(path: str, batch_size=8):
    _, _, x_test, _, _, y_test = prep_data(path)

    model, tokenizer = load_model_and_tokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    predictions = []

    for i in tqdm(range(0, len(x_test), batch_size)):
        batch = x_test.iloc[i:i + batch_size].tolist()
        inputs = tokenizer(
            batch,
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


def few_shot(path: str, k: int):
    df_train, df_val, df_test = prep_data(path)
    df_train_k = df_train.sample(n=k, random_state=42).reset_index(drop=True)
    model, tokenizer = load_model_and_tokenizer()
    train_model(model, tokenizer, df_train_k)

    return df_train_k