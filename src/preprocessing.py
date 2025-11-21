def clean_price_data(df):
    """Usuwa wiersze, gdzie nie ma ceny"""
    initial_count = len(df)

    #Operacja inplace
    df.dropna(subset=['price'], inplace=True)

    final_count = len(df)
    print(f"---CLEANING REPORT---")
    print(f"Wiersze przed: {initial_count}")
    print(f"Wiersze po: {final_count}")
    print(f"Usunieto: {initial_count - final_count} aut bez ceny")

    return df