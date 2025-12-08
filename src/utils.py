import pandas as pd
import os

def open_xlsx_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")  

    def is_utf8(s):
        try:
            s.encode("utf-8")
            return True
        except (UnicodeEncodeError, AttributeError):
            return False
    
    # Filtra linhas onde a coluna_texto é UTF-8 válida
    df = df[df.iloc[:, 0].apply(is_utf8)]
    df = df[df.iloc[:, 1].apply(is_utf8)]

    return df

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False