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
        print("--> Removed 'Unnamed: 0' column")

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

def encode_categorical_features(df):
    """
    Converts categorical text columns into numeric columns using One-Hot Encoding.
    Targets: transmission, drive, fuel, hand_drive.
    """
    print("--- ENCODING (One-Hot) ---")
    
    candidates = ['transmission', 'drive', 'fuel', 'hand_drive']
    # Only select columns that actually exist in the dataframe
    cols_to_encode = [col for col in candidates if col in df.columns]
    
    if not cols_to_encode:
        print("No columns to encode.")
        return df

    # pd.get_dummies creates new binary columns (0/1)
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    
    print(f"Encoded columns: {cols_to_encode}")
    print(f"New total columns: {len(df_encoded.columns)}")
    
    return df_encoded

def remove_outliers(df):
    """
    Removes rows with unrealistic values (outliers) to improve model quality.
    Rules:
    1.Price > 200 (200 000 JPY) - removes junk cars/down payments.
    2.Engine Capacity between 600cc and 4000cc - removes data entry errors
    """

    print("---REMOVING OUTLIERS---")
    initial_count = len(df)

    #Rule1: Price > 200
    df = df[df['price'] > 200]

    #Rule2: Engine Capacity 600cc - 4000cc
    df = df[df['engine_capacity'].between(600, 4000)]

    final_count = len(df)
    dropped_count = initial_count - final_count
    
    print(f"Rows retained: {final_count}")
    print(f"Outliers dropped: {dropped_count}")

    return df