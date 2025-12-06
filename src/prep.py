from src.utils import open_xlsx_file
from sklearn.model_selection import train_test_split
from datasets import DatasetDict
import pandas as pd

def prep_data(path: str):
    df = open_xlsx_file(path)
    df.rename(columns={df.columns[0]: "pt"}, inplace=True)
    df.rename(columns={df.columns[1]: "tp"}, inplace=True)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    x_train, x_others, y_train, y_others = train_test_split(x, y, test_size=0.3, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_others, y_others, test_size=0.15, random_state=42)
    x_train = x_train.reset_index(drop=True)
    x_val = x_val.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    x_train.to_csv("./data/x_train.csv", index=False)
    y_train.to_csv("./data/y_train.csv", index=False)

    x_val.to_csv("./data/x_val.csv", index=False)
    y_val.to_csv("./data/y_val.csv", index=False)

    x_test.to_csv("./data/x_test.csv", index=False)
    y_test.to_csv("./data/y_test.csv", index=False)

    return x_train, x_val, x_test, y_train, y_val, y_test