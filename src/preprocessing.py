import pandas as pd
import re

def clean_price_data(df):
    """
    Basic technical cleaning: Drops duplicates and missing values.
    """
    initial_count = len(df)
    
    # Drop duplicates based on link if available, otherwise strict duplicate check
    if 'link' in df.columns:
        df.drop_duplicates(subset=['link'], keep='first', inplace=True)
    else:
        df.drop_duplicates(inplace=True)
        
    # Price is the target, we cannot have it missing
    df.dropna(subset=['price'], inplace=True)
    
    # Drop index artifact if present
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print(f"Dropped: {initial_count - len(df)} rows (duplicates/empty)")
    return df

def filter_target_car(df, target_mark, target_model):
    """
    Filters the dataset for the specific car.
    (Optional now, but good for CSV cleaning)
    """
    mask = (df['mark'].str.lower() == target_mark.lower()) & \
           (df['model'].str.lower() == target_model.lower())
    
    df_filtered = df[mask].copy()
    # We drop mark/model as they are now redundant for the ML model
    df_filtered.drop(columns=['mark', 'model'], axis=1, inplace=True)
    return df_filtered

def simplify_grades(df):
    """
    Consolidates messy grade names into 'Elite' categories using Universal Keywords.
    """
    print("--- FEATURE ENGINEERING: Smart Grade Grouping ---")
    
    if 'grade' not in df.columns:
        return df

    # Universal JDM Keywords Mapping
    # Order matters: Specific high-value trims first!
    keywords = {
        'spirit r': 'Collector (Spirit R)', # RX-7 Specific
        'nur': 'Collector (Nür)',           # Skyline Specific
        'type r': 'Sport (Type R)',
        'type s': 'Sport (Type S)',
        'rs': 'Sport (RS)',
        'rz': 'Sport (RZ)',
        'sz': 'Base (SZ)',
        'wrx': 'Sport (WRX)',
        'sti': 'Sport (STI)',
        'gti': 'Sport (GTI)',
        'sir': 'Sport (SiR)',
        'aero': 'Aero/Style',
        'modulo': 'Aero/Style',
        'premium': 'Luxury',
        'luxe': 'Luxury',
        'limited': 'Luxury',
        'hybrid': 'Hybrid',
        '13g': 'Base',
        '15x': 'Base',
    }

    def map_grade(grade_text):
        text = str(grade_text).lower()
        
        for key, category in keywords.items():
            if key in text:
                return category
        
        return 'Standard/Other'

    df['grade_category'] = df['grade'].apply(map_grade)
    
    # Drop the original messy 'grade' column to prevent noise
    df.drop(columns=['grade'], inplace=True)
    
    print(f"--> Grades grouped into: {df['grade_category'].unique()}")
    return df

def remove_outliers(df):
    """
    Uses Interquartile Range (IQR) for statistical cleaning.
    """
    print("--- REMOVING OUTLIERS (Statistical IQR) ---")
    initial_count = len(df)
    
    # 1. Engine Logic (Broader range for JDM classics)
    if 'engine_capacity' in df.columns:
        df = df[df['engine_capacity'].between(600, 6000)]
    
    # 2. Price Logic (IQR Method)
    Q1 = df['price'].quantile(0.05) 
    Q3 = df['price'].quantile(0.95) 
    
    df = df[(df['price'] >= Q1) & (df['price'] <= Q3)]
    
    print(f"Rows retained: {len(df)}")
    print(f"Outliers dropped: {initial_count - len(df)}")
    
    return df

def encode_categorical_features(df):
    """
    One-Hot Encodes categorical features AND drops non-numeric metadata
    (like 'mark', 'model') that crashes XGBoost.
    """
    print("--- ENCODING (One-Hot) ---")
    
    # 1. DROP METADATA (The Fix for the ValueError)
    # XGBoost crashes if it sees 'object' columns like 'mark' or 'model'.
    cols_to_drop = ['link', 'mark', 'model', 'Unnamed: 0']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    # 2. ENCODE CATEGORIES
    candidates = ['transmission', 'drive', 'fuel', 'hand_drive', 'grade_category']
    cols_to_encode = [col for col in candidates if col in df.columns]
    
    if not cols_to_encode:
        return df

    df_encoded = pd.get_dummies(df, columns=cols_to_encode, dtype=int)
    print(f"Encoded columns: {cols_to_encode}")
    return df_encoded