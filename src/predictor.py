import pandas as pd
import numpy as np
import warnings
import joblib
from config import TWEEDIE_POWER
from pipeline import FireFeatureEngineeringPipeline

warnings.filterwarnings('ignore')

class FireRiskPredictor:
    def __init__(self):
        self.pipeline = FireFeatureEngineeringPipeline()
        self.model = None

    def _load_model(self):
        if self.model is None:
            # Artık çok hafif olan SVR modelini yüklüyoruz
            self.model = joblib.load('models/sklearn_node.pkl')
        return self.model

    def predict(self, raw_data: dict) -> tuple:
        df = pd.DataFrame([raw_data])
        processed_df = self.pipeline.transform(df)
        
        model = self._load_model()
        
        y_pred_tweedie = model.predict(processed_df.values)[0]
        
        margin = getattr(model, 'margin_raw_', 6.92)
        
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
        print(f"INFO: Tahmin: {tahmin} Hektar (± {marj} ha Güvenle)")
    except Exception as e:
        print(f"ERROR: Model yüklenemedi: {e}")
