import pandas as pd

def clean_price_data(df):
    """
    Basic technical cleaning:
    1. Removes rows with missing price.
    2. Removes the redundant 'Unnamed: 0' column if present.
    """
    initial_count = len(df)
    
    # STEP 1: Remove rows where price is NaN
    df.dropna(subset=['price'], inplace=True)
    
    # STEP 2: Remove the artifact index column
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    final_count = len(df)
    print(f"--- CLEANING REPORT ---")
    print(f"Rows before: {initial_count}")
    print(f"Rows after:  {final_count}")
    print(f"Dropped:     {initial_count - final_count} rows without price")

    return df

def filter_target_car(df, target_mark, target_model):
    """
    Filters the dataset to keep only the specific car make and model.
    Removes 'mark' and 'model' columns afterwards (Zero Variance).
    """
    print(f"--- FILTERING: {target_mark} {target_model} ---")
    
    # 1. Create mask (case insensitive)
    mask = (df['mark'] == target_mark.lower()) & (df['model'] == target_model.lower())
    
    # 2. Apply mask and create a copy
    df_filtered = df[mask].copy()
    
    # 3. Drop redundant columns
    cols_to_drop = ['mark', 'model']
    df_filtered.drop(columns=cols_to_drop, axis=1, inplace=True)
    
    print(f"Found matching cars: {len(df_filtered)}")
    
    return df_filtered

def remove_outliers(df):
    """
    Removes rows with unrealistic values (outliers).
    Rules:
    1. Price > 50 ('000 JPY) -> 50,000 JPY minimum (covers both synthetic/live scales)
    2. Engine Capacity 600cc - 4000cc
    """
    print("--- REMOVING OUTLIERS ---")
    initial_count = len(df)
    
    # Rule 1: Price > 50 (50,000 JPY)
    df = df[df['price'] > 50]
    
    # Rule 2: Engine Capacity
    df = df[df['engine_capacity'].between(600, 4000)]
    
    final_count = len(df)
    dropped_count = initial_count - final_count
    
    print(f"Rows retained: {final_count}")
    print(f"Outliers dropped: {dropped_count}")
    
    return df

def encode_categorical_features(df):
    """
    Converts categorical text columns into numeric columns using One-Hot Encoding.
    CRITICAL: Drops non-numeric columns that are not useful for training (like 'link').
    """
    print("--- ENCODING (One-Hot) ---")
    
    # 1. DROP NON-TRAINING COLUMNS (Crucial Fix for ValueError)
    if 'link' in df.columns:
        df = df.drop(columns=['link'])
        print("--> Dropped 'link' column (not for training)")
    
    # 2. ENCODE CATEGORICALS
    candidates = ['transmission', 'drive', 'fuel', 'hand_drive']
    cols_to_encode = [col for col in candidates if col in df.columns]
    
    if not cols_to_encode:
        print("No columns to encode.")
        return df

    df_encoded = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    
    print(f"Encoded columns: {cols_to_encode}")
    print(f"New total columns: {len(df_encoded.columns)}")
    
    return df_encoded