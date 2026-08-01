import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.metrics import mean_squared_error
import warnings

from config import (
    TARGET_COL, BASE_DIR, MODEL_DIR,
    COLLECTED_DATA_PATH, PROCESSED_DATA_PATH,
    CHAMPION_MODEL_PATH
)
from pipeline import FireFeatureEngineeringPipeline
from model_trainer import inverse_transform, train_svr_model

warnings.filterwarnings("ignore")

BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "holdout", "benchmark_test.csv")
OPTUNA_DB = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_study.db')}"
STUDY_NAME = "SVR_Retrain_Optimization"

SEPARATOR = "-" * 60
HEADER = "=" * 60


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}]  {msg}")


def load_training_data():
    train_df = pd.read_csv(PROCESSED_DATA_PATH)
    log(f"Base training data loaded: {len(train_df)} rows")

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


def run_retrain():
    print(f"\n{HEADER}")
    print(f"  MLOPS SVR RETRAIN PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(HEADER)

    train_df = load_training_data()
    bench_df, y_bench = load_benchmark()
    X_bench = bench_df.drop(columns=[TARGET_COL], errors='ignore').values

    print(SEPARATOR)

    # Mevcut şampiyonu değerlendir
    current_rmse = None
    if os.path.exists(CHAMPION_MODEL_PATH):
        log("Evaluating current production SVR model...")
        current_model = joblib.load(CHAMPION_MODEL_PATH)
        current_rmse = evaluate_model(current_model, X_bench, y_bench)
        log(f"Current model RMSE:   {current_rmse:.4f}")
    else:
        log("No production model found. Training initial model.")

    print(SEPARATOR)

    # Aday modeli eğit (ortak fonksiyonu kullan)
    X_train = train_df.drop(columns=[TARGET_COL]).values
    y_train = train_df[TARGET_COL].values

    log("Training candidate SVR model (Optuna, 15 trials)...")
    candidate = train_svr_model(X_train, y_train, OPTUNA_DB, STUDY_NAME, n_trials=15)
    log(f"Candidate margin: ±{candidate.margin_raw_:.2f} ha")

    candidate_rmse = evaluate_model(candidate, X_bench, y_bench)
    log(f"Candidate model RMSE: {candidate_rmse:.4f}")

    print(SEPARATOR)

    # Champion vs Challenger
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