import pandas as pd

def open_xlsx_file(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")  
    return df

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False