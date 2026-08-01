import os
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error

from pytorch_tabular import TabularModel
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

import optuna
import mlflow
import warnings

from config import (
    CATEGORICAL_COLS, CONTINUOUS_COLS, TARGET_COL,
    CHAMPION_MODEL_PATH, TWEEDIE_P, TWEEDIE_POWER,
    BASE_DIR, MODEL_DIR, COLLECTED_DATA_PATH, PROCESSED_DATA_PATH
)
from pipeline import FireFeatureEngineeringPipeline

warnings.filterwarnings("ignore")

BENCHMARK_PATH = os.path.join(BASE_DIR, "data", "holdout", "benchmark_test.csv")
ARCHIVE_DIR = os.path.join(MODEL_DIR, "archive")
OPTUNA_DB = f"sqlite:///{os.path.join(BASE_DIR, 'optuna_study.db')}"
STUDY_NAME = "NODE_Optimization"

SEPARATOR = "-" * 60
HEADER = "=" * 60


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}]  {msg}")


def load_training_data():
    train_df = pd.read_csv(PROCESSED_DATA_PATH)
    log(f"Training data loaded: {len(train_df)} rows")

    if os.path.exists(COLLECTED_DATA_PATH) and os.path.getsize(COLLECTED_DATA_PATH) > 0:
        new_raw = pd.read_csv(COLLECTED_DATA_PATH)
        pipeline = FireFeatureEngineeringPipeline()
        new_processed = pipeline.transform(new_raw)
        train_df = pd.concat([train_df, new_processed], ignore_index=True)
        log(f"New data merged. Total: {len(train_df)} rows")
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


def evaluate_model(model, bench_df, y_true):
    pred_df = model.predict(bench_df)
    pred_col = [c for c in pred_df.columns if "prediction" in c.lower() or c == TARGET_COL][0]
    y_pred_tweedie = pred_df[pred_col].values

    y_pred_hectares = np.where(
        y_pred_tweedie <= 0, 0.0,
        (y_pred_tweedie * TWEEDIE_POWER) ** (1.0 / TWEEDIE_POWER)
    )
    y_pred_hectares = np.where(y_pred_hectares < 0.01, 0.0, y_pred_hectares)

    return np.sqrt(mean_squared_error(y_true, y_pred_hectares))


def get_best_params():
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=OPTUNA_DB)
        best = study.best_params
        log(f"Hyperparameters loaded from Optuna DB (best trial #{study.best_trial.number})")
        return best
    except Exception:
        log("Optuna DB not found. Using default hyperparameters.")
        return {"num_layers": 4, "num_trees": 512, "depth": 4, "learning_rate": 0.001}


def train_candidate(train_df, params):
    data_config = DataConfig(
        target=[TARGET_COL],
        continuous_cols=CONTINUOUS_COLS,
        categorical_cols=CATEGORICAL_COLS
    )
    model_config = NodeConfig(
        task="regression",
        loss="MSELoss",
        metrics=["mean_squared_error"],
        num_layers=params["num_layers"],
        num_trees=params["num_trees"],
        depth=params["depth"],
        learning_rate=params["learning_rate"]
    )
    trainer_config = TrainerConfig(
        batch_size=128,
        max_epochs=100,
        early_stopping="valid_loss",
        early_stopping_patience=20,
        early_stopping_mode="min",
        load_best=True,
        trainer_kwargs={"enable_model_summary": False}
    )

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=OptimizerConfig(),
        trainer_config=trainer_config
    )
    model.fit(train=train_df)
    return model


def archive_current_model():
    if not os.path.exists(CHAMPION_MODEL_PATH):
        return
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
    archive_path = os.path.join(ARCHIVE_DIR, f"model_{timestamp}")
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    shutil.copytree(CHAMPION_MODEL_PATH, archive_path)
    log(f"Previous model archived: {archive_path}")


def run_retrain():
    print(f"\n{HEADER}")
    print(f"  RETRAIN PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(HEADER)

    train_df = load_training_data()
    bench_df, y_bench = load_benchmark()

    print(SEPARATOR)

    current_rmse = None
    if os.path.exists(CHAMPION_MODEL_PATH):
        log("Evaluating current production model...")
        current_model = TabularModel.load_model(CHAMPION_MODEL_PATH)
        current_rmse = evaluate_model(current_model, bench_df, y_bench)
        log(f"Current model RMSE:   {current_rmse:.4f}")
    else:
        log("No production model found. Training initial model.")

    print(SEPARATOR)

    params = get_best_params()
    log("Training candidate model...")
    candidate = train_candidate(train_df, params)
    candidate_rmse = evaluate_model(candidate, bench_df, y_bench)
    log(f"Candidate model RMSE: {candidate_rmse:.4f}")

    print(SEPARATOR)

    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR, 'mlops_tracking.db')}")
    mlflow.set_experiment("Retrain_Pipeline")

    with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params(params)
        mlflow.log_metric("candidate_rmse", candidate_rmse)

        if current_rmse is not None:
            mlflow.log_metric("current_rmse", current_rmse)
            mlflow.log_metric("delta_rmse", current_rmse - candidate_rmse)

        promoted = current_rmse is None or candidate_rmse < current_rmse
        mlflow.log_metric("promoted", int(promoted))

        if promoted:
            archive_current_model()
            if os.path.exists(CHAMPION_MODEL_PATH):
                shutil.rmtree(CHAMPION_MODEL_PATH)
            os.makedirs(os.path.dirname(CHAMPION_MODEL_PATH), exist_ok=True)
            candidate.save_model(CHAMPION_MODEL_PATH)

            if current_rmse is not None:
                log(f"RESULT: Candidate promoted. RMSE improved {current_rmse:.4f} -> {candidate_rmse:.4f}")
            else:
                log(f"RESULT: Initial model registered. RMSE: {candidate_rmse:.4f}")
            log(f"Model saved: {CHAMPION_MODEL_PATH}")
        else:
            log(f"RESULT: Current model retained. (Current: {current_rmse:.4f} | Candidate: {candidate_rmse:.4f})")

    print(HEADER)
    print(f"  RETRAIN PIPELINE COMPLETED")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{HEADER}\n")


if __name__ == "__main__":
    run_retrain()