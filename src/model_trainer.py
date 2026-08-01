import os
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from config import TARGET_COL, TWEEDIE_POWER


def inverse_transform(y_tweedie):
    """Tweedie uzayından orijinal hektar değerine geri dönüşüm."""
    return np.power(np.maximum(y_tweedie, 0) * TWEEDIE_POWER, 1/TWEEDIE_POWER)


def objective(trial, X, y):
    """Optuna Bayesian Optimization için amaç fonksiyonu."""
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


def train_svr_model(X, y_raw, optuna_db, study_name, n_trials=50):
    """
    SVR modelini Optuna ile optimize edip eğitir.
    
    Args:
        X: Öznitelik matrisi (numpy array)
        y_raw: Ham hedef değerler (hektar)
        optuna_db: Optuna SQLite bağlantı URI'si
        study_name: Optuna çalışma adı
        n_trials: Optimizasyon deneme sayısı
    
    Returns:
        Eğitilmiş Pipeline nesnesi (margin_raw_ özniteliği eklenmiş)
    """
    y_tweedie = (np.power(y_raw, TWEEDIE_POWER)) / TWEEDIE_POWER

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=optuna_db,
        load_if_exists=True
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(lambda trial: objective(trial, X, y_tweedie), n_trials=n_trials)

    best_model = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', **study.best_params))
    ])
    best_model.fit(X, y_tweedie)

    # %90 güven aralığı marjı hesapla
    preds_tweedie = best_model.predict(X)
    preds_raw = inverse_transform(preds_tweedie)
    residuals_raw = np.abs(y_raw - preds_raw)
    best_model.margin_raw_ = float(np.quantile(residuals_raw, 0.90))

    return best_model


def train_and_save():

    print("INFO: Loading dataset [processed_forestfires_v2.csv]")
    df = pd.read_csv("data/processed/processed_forestfires_v2.csv")

    X = df.drop(columns=[TARGET_COL]).values
    y = df[TARGET_COL].values

    print("INFO: Initializing Bayesian Optimization via Optuna.")
    os.makedirs("models", exist_ok=True)

    model = train_svr_model(
        X, y,
        optuna_db="sqlite:///models/optuna_study.db",
        study_name="svr_fire_risk_v2",
        n_trials=50
    )

    print(f"INFO: Final Margin computed as ±{model.margin_raw_:.2f} ha.")

    joblib.dump(model, "models/sklearn_node.pkl")
    print("INFO: Model successfully saved to [models/sklearn_node.pkl].")


if __name__ == '__main__':
    train_and_save()