import pandas as pd
from sklearn.model_selection import train_test_split
# NEW IMPORT: The Heavy Artillery
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def split_data(df):
    """
    Splits the DataFrame into features (X) and target (y),
    and then into training and testing sets.
    """
    print("--- DATA SPLITTING ---")
    target_column = 'price'
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Initializes and trains the Random Forest Regressor.
    """
    print("--- TRAINING MODEL (Random Forest) ---")
    
    # n_estimators=100 -> We create 100 trees (the "Forest")
    # random_state=42  -> Ensures reproducible results
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # The syntax is identical to LinearRegression!
    model.fit(X_train, y_train)
    
    print("--> Model trained successfully.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Predicts on the test set and calculates performance metrics.
    """
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    return mae, r2, predictions