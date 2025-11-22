import pandas as pd
from pathlib import Path
import numpy as np

def load_raw_data(filepath):
    """
    Loads raw data from a CSV file.
    
    Args:
        filepath (Path): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data.
    """
    print(f"... Loading data from: {filepath}")
    return pd.read_csv(filepath)

def save_processed_data(df, filepath):
    """
    Saves the DataFrame to a CSV file without the index.
    
    Args:
        df (pd.DataFrame): Data to save.
        filepath (Path): Destination path.
    """
    # Create parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV, index=False prevents writing row numbers
    df.to_csv(filepath, index=False)
    
    print(f"--> Success: Data saved to: {filepath}")

def generate_synthetic_data(n_samples = 1000):
    """
    Generates a synthetic dataset for Honda Fit.
    Logic: Price decreases with Age and Mileage

    Args:
        n_samples (int): Number of rows to generate.

    Returns:
        pd.DataFrame: Synthetic dataset.
    """
    
    print(f"---GENERATING SYNTHETIC DATA ({n_samples} samples)")

    #Set seed for reproducibility
    np.random.seed(42)
    
    #Generate Features
    years = np.random.randint(2005, 2024, n_samples)

    mileage = np.random.randint(5000, 200000, n_samples)
    
    # Engine: Mostly 1300cc or 1500cc
    engines = np.random.choice([1300, 1500], n_samples)
    
    # Transmission: 70% AT, 30% MT
    transmissions = np.random.choice(['at', 'mt'], n_samples, p=[0.7, 0.3])
    
    # Drive: 80% 2wd, 20% 4wd
    drives = np.random.choice(['2wd', '4wd'], n_samples, p=[0.8, 0.2])

    #Logic
    base_price = 2500
    current_year = 2025
    age = current_year - years
    
    #Depreciation rules:
    #Lose 100k JPY per year
    #Lose 5 JPY per km
    price = base_price - (age * 100) - (mileage * 0.005)

    #Adding random noise
    noise = np.random.normal(0, 200, n_samples)
    price = price + noise

    #Ensure no negative prices
    price = np.maximum(price, 100)

    #Create dataframe
    df = pd.DataFrame({
        'price': price.astype(int),
        'year': years,
        'mileage': mileage,
        'engine_capacity': engines,
        'transmission': transmissions,
        'drive': drives,
        'hand_drive': 'rhd',      # Static
        'fuel': 'gasoline',       # Static
        'mark': 'honda',          # Static (for filter compatibility)
        'model': 'fit'            # Static (for filter compatibility)
    })
    
    print("--> Synthetic Data Generated Successfully.")
    return df