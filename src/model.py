import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from pathlib import Path

def split_data(df):
    print("--- DATA SPLITTING ---")
    target_column = 'price'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    print("--- TRAINING MODEL (Grid Search Optimization) ---")
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    print(f"--> Best Parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    return metrics, predictions

def calculate_advanced_metrics(model, X_train, y_train, X_test, y_test, predictions):
    """
    Calculates diagnostics to check for overfitting and error patterns.
    """
    # 1. Overfitting Check (Train R2 vs Test R2)
    train_predictions = model.predict(X_train)
    train_r2 = r2_score(y_train, train_predictions)
    
    # 2. Residuals (Real - Predicted)
    residuals = y_test - predictions
    
    return {
        "train_r2": train_r2,
        "residuals": residuals
    }

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    return feature_importance_df

def save_model(model, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"--> Model saved to: {filepath}")