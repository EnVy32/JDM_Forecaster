#main.py

#Import biblioteki do zarządzania scieżkami
from pathlib import Path

#Import naszych wlasnych funkcji z folderu scr 
from src.data_loader import load_raw_data
from src.preprocessing import clean_price_data

def main():
    print("---[JDM_Forecaster] Start Procesu---")

    #Definiowanie ścieżek niezależnie od systemu operacyjnego
    #Path.cwd() to "Current Working Directory" 
    project_root = Path.cwd()

    #Operator '/' w pathlib laczy elementy ścieżki w bezpieczny sposób 
    data_path = project_root / 'data' / 'raw' / 'cars_datasets.csv'

    #Weryfikacja czy plik istnieje
    if not data_path.exists():
        print(f"Error: Nie znaleziono pliku pod adresem: {data_path}")
        return
    

    #Wczytywanie zaimportowanych funkcji
    df = load_raw_data(data_path)

    #Czyszczenie
    df_cleaned = clean_price_data(df)

    print("---[JDM_Forecaster] Proces Zakończony sukcesem---")

if __name__ == "__main__":
    main()