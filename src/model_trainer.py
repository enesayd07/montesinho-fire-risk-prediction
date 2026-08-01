import logging
import os
import sys
import pandas as pd
import numpy as np
import joblib
from contextlib import contextmanager
from sklearn.base import BaseEstimator, RegressorMixin
from pytorch_tabular import TabularModel
from pytorch_tabular.models import NodeConfig
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig

from config import TARGET_COL, TWEEDIE_POWER

@contextmanager
def suppress_output():
    logger = logging.getLogger("pytorch_tabular")
    pl_logger = logging.getLogger("lightning.pytorch")
    old_level = logger.level
    pl_old_level = pl_logger.level
    logger.setLevel(logging.WARNING)
    pl_logger.setLevel(logging.WARNING)
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            logger.setLevel(old_level)
            pl_logger.setLevel(pl_old_level)

class HurdleNODE(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.classifier = None
        self.regressor = None
        self.margin_raw_ = 0.0

    def fit(self, X_cls, y_cls_raw, X_reg, y_reg_tw, X_val=None, y_val_tw=None):
        # 1. CLASSIFIER (Augmented data uzerinde Egitilir)
        y_cls = (y_cls_raw > 0).astype(int)
        train_df_cls = pd.DataFrame(X_cls).copy()
        train_df_cls.columns = [f"col_{i}" for i in range(train_df_cls.shape[1])]
        train_df_cls["target"] = y_cls
        
        val_df_cls = None
        if X_val is not None and y_val_tw is not None:
            val_df_cls = pd.DataFrame(X_val).copy()
            val_df_cls.columns = [f"col_{i}" for i in range(val_df_cls.shape[1])]
            val_df_cls["target"] = (y_val_tw > 0).astype(int)

        print("-> [Asama 1/2] Siniflandirma (Classifier) Agi Egitiliyor (Augmented Data ile)...")
        data_config_cls = DataConfig(target=["target"], continuous_cols=list(train_df_cls.columns[:-1]), categorical_cols=[])
        trainer_config_cls = TrainerConfig(batch_size=32, max_epochs=20, early_stopping="valid_loss", early_stopping_patience=5, progress_bar="none", trainer_kwargs={"enable_model_summary": False})
        model_config_cls = NodeConfig(task="classification", num_layers=2, num_trees=512, depth=4)
        
        self.classifier = TabularModel(
            data_config=data_config_cls, model_config=model_config_cls, 
            optimizer_config=OptimizerConfig(), trainer_config=trainer_config_cls, 
            verbose=False, suppress_lightning_logger=True
        )
        
        with suppress_output():
            if len(train_df_cls) % 32 == 1: train_df_cls = train_df_cls.iloc[:-1]
            self.classifier.fit(train_df_cls, val_df_cls)

        # 2. REGRESSOR (Saf Orijinal Data uzerinde Egitilir!)
        print("-> [Asama 2/2] Regresyon (Regressor) Agi Egitiliyor (Orijinal Saf Data ile)...")
        pos_mask = y_reg_tw > 0
        X_reg_pos = X_reg[pos_mask]
        y_reg_pos = y_reg_tw[pos_mask]
        
        train_df_reg = pd.DataFrame(X_reg_pos).copy()
        train_df_reg.columns = [f"col_{i}" for i in range(train_df_reg.shape[1])]
        train_df_reg["target"] = y_reg_pos

        val_df_reg = None
        if X_val is not None and y_val_tw is not None:
            pos_mask_val = y_val_tw > 0
            if pos_mask_val.sum() > 0:
                val_df_reg = pd.DataFrame(X_val[pos_mask_val]).copy()
                val_df_reg.columns = [f"col_{i}" for i in range(val_df_reg.shape[1])]
                val_df_reg["target"] = y_val_tw[pos_mask_val]

        data_config_reg = DataConfig(target=["target"], continuous_cols=list(train_df_reg.columns[:-1]), categorical_cols=[])
        trainer_config_reg = TrainerConfig(batch_size=32, max_epochs=20, early_stopping="valid_loss", early_stopping_patience=5, progress_bar="none", trainer_kwargs={"enable_model_summary": False})
        model_config_reg = NodeConfig(task="regression", num_layers=2, num_trees=512, depth=4, metrics=["mean_squared_error"])
        
        self.regressor = TabularModel(
            data_config=data_config_reg, model_config=model_config_reg, 
            optimizer_config=OptimizerConfig(), trainer_config=trainer_config_reg, 
            verbose=False, suppress_lightning_logger=True
        )
        
        with suppress_output():
            if len(train_df_reg) % 32 == 1: train_df_reg = train_df_reg.iloc[:-1]
            self.regressor.fit(train_df_reg, val_df_reg)
            
        return self

    def predict(self, X):
        test_df = pd.DataFrame(X).copy()
        test_df.columns = [f"col_{i}" for i in range(test_df.shape[1])]
        
        with suppress_output():
            cls_preds = self.classifier.predict(test_df)
        pred_col_cls = [col for col in cls_preds.columns if "prediction" in col.lower()][0]
        is_fire = cls_preds[pred_col_cls].values
        
        with suppress_output():
            reg_preds = self.regressor.predict(test_df)
        pred_col_reg = [col for col in reg_preds.columns if "prediction" in col.lower()][0]
        fire_amount = reg_preds[pred_col_reg].values
        
        return is_fire * fire_amount

def inverse_transform(y_tweedie):
    # Senin gonderdigin kodla Birebir ayni guvenli transformasyon (np.maximum)
    return np.power(np.maximum(y_tweedie, 0) * TWEEDIE_POWER, 1/TWEEDIE_POWER)

def train_and_save():
    print("Siniflandirici icin EFSANEVI V2 Veri Seti (Augmented) Yukleniyor...")
    X_train_cls = pd.read_csv("data/processed/X_train_augmented_v2.csv").values
    y_train_cls = pd.read_csv("data/processed/y_train_augmented_v2.csv").values.flatten()
    
    print("Regresyon icin SAF ORIJINAL V2 Veri Seti Yukleniyor...")
    X_train_reg = pd.read_csv("data/processed/X_train_v2.csv").values
    y_train_reg = pd.read_csv("data/processed/y_train_v2.csv").values.flatten()
    
    X_val = pd.read_csv("data/processed/X_test_v2.csv").values
    y_val = pd.read_csv("data/processed/y_test_v2.csv").values.flatten()
    
    # Sadece Regresyon agina girecek olan saf veri Tweedie'ye donusturuluyor
    y_train_reg_tw = (np.power(y_train_reg, TWEEDIE_POWER)) / TWEEDIE_POWER
    y_val_tw = (np.power(y_val, TWEEDIE_POWER)) / TWEEDIE_POWER
    
    hurdle_model = HurdleNODE(random_state=42)
    hurdle_model.fit(X_train_cls, y_train_cls, X_train_reg, y_train_reg_tw, X_val, y_val_tw)
    
    print("-> Hata payi (margin) V2 Test Setinden hesaplaniyor...")
    preds_val_tweedie = hurdle_model.predict(X_val) 
    preds_val_raw = inverse_transform(preds_val_tweedie)
    
    residuals_raw = np.abs(y_val - preds_val_raw)
    margin_raw = float(np.quantile(residuals_raw, 0.90))
    hurdle_model.margin_raw_ = margin_raw
    
    print(f"-> Egitim Bitti! %90 Guven Araligi (Margin): ±{margin_raw:.2f} ha")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(hurdle_model, "models/sklearn_node.pkl")
    print("-> Hurdle Model basariyla 'models/sklearn_node.pkl' olarak kaydedildi!")

if __name__ == '__main__':
    train_and_save()
