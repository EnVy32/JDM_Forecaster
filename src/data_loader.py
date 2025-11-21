import pandas as pd

def load_raw_data(filepath):
    """Wczytuje surowe dane z pliku CSV"""
    print(f"... Wczytuje dane z : {filepath}")
    return pd.read_csv(filepath)