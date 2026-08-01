import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import optuna
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

from config import (
    TARGET_COL, TWEEDIE_POWER,
    BASE_DIR, MODEL_DIR, COLLECTED_DATA_PATH, PROCESSED_DATA_PATH
)
from pipeline import FireFeatureEngineeringPipeline

warnings.filterwarnings("ignore")

BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "holdout", "benchmark_test.csv")
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "sklearn_node.pkl")
OPTUNA_DB = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_study.db')}"
STUDY_NAME = "SVR_Retrain_Optimization"

SEPARATOR = "-" * 60
HEADER = "=" * 60

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}]  {msg}")

def inverse_transform(y_tweedie):
    return np.power(np.maximum(y_tweedie, 0) * TWEEDIE_POWER, 1/TWEEDIE_POWER)

def load_training_data():
    train_df = pd.read_csv(PROCESSED_DATA_PATH)
    log(f"Base training data loaded: {len(train_df)} rows")

    # Yeni toplanan veri varsa (API'den gelenler), ana veriyle birleştir
    if os.path.exists(COLLECTED_DATA_PATH) and os.path.getsize(COLLECTED_DATA_PATH) > 0:
        new_raw = pd.read_csv(COLLECTED_DATA_PATH)
        pipeline = FireFeatureEngineeringPipeline()
        new_processed = pipeline.transform(new_raw)
        train_df = pd.concat([train_df, new_processed], ignore_index=True)
        log(f"New reported data merged. Total: {len(train_df)} rows")
    else:
        log("No new data found. Proceeding with existing dataset.")

    return train_df

def load_benchmark():
    bench_raw = pd.read_csv(BENCHMARK_PATH)
    y_bench = bench_raw[TARGET_COL].values.copy()
    pipeline = FireFeatureEngineeringPipeline()
    bench_processed = pipeline.transform(bench_raw)
    log(f"Benchmark set loaded: {len(bench_raw)} rows (holdout)")
    return bench_processed, y_bench

def evaluate_model(model, X_bench, y_true):
    preds_tweedie = model.predict(X_bench)
    preds_raw = inverse_transform(preds_tweedie)
    return np.sqrt(mean_squared_error(y_true, preds_raw))

def objective(trial, X, y):
    C = trial.suggest_float('C', 0.01, 100.0, log=True)
    epsilon = trial.suggest_float('epsilon', 0.001, 1.0, log=True)
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))
    ])
    
    from sklearn.model_selection import KFold, cross_val_score
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
    return -scores.mean()

def train_candidate(train_df):
    X = train_df.drop(columns=[TARGET_COL]).values
    y = train_df[TARGET_COL].values
    y_tweedie = (np.power(y, TWEEDIE_POWER)) / TWEEDIE_POWER

    study = optuna.create_study(
        study_name=STUDY_NAME, 
        direction="minimize", 
        storage=OPTUNA_DB, 
        load_if_exists=True
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    log("Running Optuna Bayesian Optimization (15 trials) for SVR...")
    study.optimize(lambda trial: objective(trial, X, y_tweedie), n_trials=15)
    
    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', **study.best_params))
    ])
    best_model.fit(X, y_tweedie)
    
    preds_tweedie = best_model.predict(X) 
    preds_raw = inverse_transform(preds_tweedie)
    residuals_raw = np.abs(y - preds_raw)
    best_model.margin_raw_ = float(np.quantile(residuals_raw, 0.90))
    
    log(f"Candidate model trained. Margin: ±{best_model.margin_raw_:.2f} ha")
    return best_model

def run_retrain():
    print(f"\n{HEADER}")
    print(f"  MLOPS SVR RETRAIN PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(HEADER)

    train_df = load_training_data()
    bench_df, y_bench = load_benchmark()
    X_bench = bench_df.drop(columns=[TARGET_COL], errors='ignore').values

    print(SEPARATOR)

    current_rmse = None
    if os.path.exists(CHAMPION_MODEL_PATH):
        log("Evaluating current production SVR model...")
        current_model = joblib.load(CHAMPION_MODEL_PATH)
        current_rmse = evaluate_model(current_model, X_bench, y_bench)
        log(f"Current model RMSE:   {current_rmse:.4f}")
    else:
        log("No production model found. Training initial model.")

    print(SEPARATOR)

    candidate = train_candidate(train_df)
    candidate_rmse = evaluate_model(candidate, X_bench, y_bench)
    log(f"Candidate model RMSE: {candidate_rmse:.4f}")

    print(SEPARATOR)

    promoted = current_rmse is None or candidate_rmse < current_rmse

    if promoted:
        if os.path.exists(CHAMPION_MODEL_PATH):
            archive_dir = os.path.join(MODEL_DIR, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            shutil.copy2(CHAMPION_MODEL_PATH, os.path.join(archive_dir, f"sklearn_node_{timestamp}.pkl"))
        
        joblib.dump(candidate, CHAMPION_MODEL_PATH)
        
        if current_rmse is not None:
            log(f"RESULT: Candidate PROMOTED. RMSE improved {current_rmse:.4f} -> {candidate_rmse:.4f}")
        else:
            log(f"RESULT: Initial model registered. RMSE: {candidate_rmse:.4f}")
        log(f"Model saved: {CHAMPION_MODEL_PATH}")
    else:
        log(f"RESULT: Current model RETAINED. (Current: {current_rmse:.4f} | Candidate: {candidate_rmse:.4f})")

    print(HEADER)
    print(f"  RETRAIN PIPELINE COMPLETED")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{HEADER}\n")

if __name__ == "__main__":
    run_retrain()