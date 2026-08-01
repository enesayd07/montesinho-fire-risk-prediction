import pandas as pd
import numpy as np
import warnings
import joblib
from config import TWEEDIE_POWER
from pipeline import FireFeatureEngineeringPipeline

# Modeli joblib ile cekerken taninmasi icin class'i import ediyoruz
from model_trainer import HurdleNODE
import __main__
__main__.HurdleNODE = HurdleNODE

warnings.filterwarnings('ignore')

class FireRiskPredictor:
    def __init__(self):
        self.pipeline = FireFeatureEngineeringPipeline()
        self.model = None

    def _load_model(self):
        if self.model is None:
            # Optuna modeli yerine senin 17.1M parametreli NODE modelini yukluyoruz
            self.model = joblib.load('models/sklearn_node.pkl')
        return self.model

    def predict(self, raw_data: dict) -> tuple:
        df = pd.DataFrame([raw_data])
        processed_df = self.pipeline.transform(df)
        
        model = self._load_model()
        
        # Modelin icine .values olarak veriyoruz
        y_pred_tweedie = model.predict(processed_df.values)[0]
        
        # Model egitilirken icine gomdugumuz margin degerini cekiyoruz
        margin = getattr(model, 'margin_raw_', 9.16)
        
        if y_pred_tweedie <= 0:
            return 0.0, margin
            
        y_raw_hectares = (y_pred_tweedie * TWEEDIE_POWER) ** (1.0 / TWEEDIE_POWER)
        
        if y_raw_hectares < 0.01:
            y_raw_hectares = 0.0
            
        return round(y_raw_hectares, 4), round(margin, 2)

if __name__ == '__main__':
    sample_input = {
        'X': 7, 'Y': 5, 'month': 'aug', 'day': 'fri',
        'FFMC': 96.1, 'DMC': 181.1, 'DC': 671.2, 'ISI': 14.3,
        'temp': 28.7, 'RH': 30, 'wind': 4.5, 'rain': 0.0
    }
    
    try:
        predictor = FireRiskPredictor()
        tahmin, marj = predictor.predict(sample_input)
        print(f"Tahmin: {tahmin} Hektar (± {marj} ha Guvenle)")
    except Exception as e:
        print(f"Model yuklenemedi: {e}")
