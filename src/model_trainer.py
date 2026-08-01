import os
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from config import TARGET_COL, TWEEDIE_POWER

def inverse_transform(y_tweedie):
    return np.power(np.maximum(y_tweedie, 0) * TWEEDIE_POWER, 1/TWEEDIE_POWER)

def objective(trial, X, y):
    C = trial.suggest_float('C', 0.01, 100.0, log=True)
    epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))
    ])
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    
    return -scores.mean()

def train_and_save():
    print("INFO: Loading dataset [processed_forestfires_v2.csv]")
    df = pd.read_csv("data/processed/processed_forestfires_v2.csv")
    
    X = df.drop(columns=['area']).values
    y = df['area'].values
    y_tweedie = (np.power(y, TWEEDIE_POWER)) / TWEEDIE_POWER
    
    print("INFO: Initializing Bayesian Optimization via Optuna.")
    os.makedirs("models", exist_ok=True)
    
    study = optuna.create_study(
        study_name="svr_fire_risk_v2", 
        direction="minimize", 
        storage="sqlite:///models/optuna_study.db", 
        load_if_exists=True
    )
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda trial: objective(trial, X, y_tweedie), n_trials=50)
    
    print(f"INFO: Optimization completed. Best MSE: {study.best_value:.4f}")
    
    print("INFO: Training final SVR model with optimal parameters.")
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', **study.best_params))
    ])
    
    best_model.fit(X, y_tweedie)
    
    print("INFO: Calculating 90% confidence margin.")
    preds_tweedie = best_model.predict(X) 
    preds_raw = inverse_transform(preds_tweedie)
    
    residuals_raw = np.abs(y - preds_raw)
    margin_raw = float(np.quantile(residuals_raw, 0.90))
    best_model.margin_raw_ = margin_raw
    
    print(f"INFO: Final Margin computed as ±{margin_raw:.2f} ha.")
    
    joblib.dump(best_model, "models/sklearn_node.pkl")
    print("INFO: Model successfully saved to [models/sklearn_node.pkl].")

if __name__ == '__main__':
    train_and_save()