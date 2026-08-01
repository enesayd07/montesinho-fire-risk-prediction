import pandas as pd
import numpy as np

class FireFeatureEngineeringPipeline:
    def __init__(self):
        # Aylar ve gunler icin sozlukler
        self.months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                       'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
        self.days = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        # 1. Ay ve Gun Donusumleri
        data['month_num'] = data['month'].map(self.months)
        data['day_num'] = data['day'].map(self.days)

        # 2. Dongusel Kodlama 
        data['month_sin'] = np.sin(2 * np.pi * data['month_num'] / 12.0)
        data['month_cos'] = np.cos(2 * np.pi * data['month_num'] / 12.0)
        data['day_sin'] = np.sin(2 * np.pi * data['day_num'] / 7.0)
        data['day_cos'] = np.cos(2 * np.pi * data['day_num'] / 7.0)

        # 3. Insan Faktoru ve Mevsimsellik 
        data['is_weekend'] = data['day'].isin(['sat', 'sun']).astype(int)
        peak_months = ['mar', 'aug', 'sep', 'dec']
        data['is_peak_season'] = data['month'].isin(peak_months).astype(int)

        # 4. Meteorolojik Tehlike Gostergeleri 
        data['moderate_wind_danger'] = ((data['wind'] >= 3.5) & (data['wind'] <= 6.0)).astype(int)
        data['hot_and_dry'] = ((data['temp'] >= 20.0) & (data['RH'] <= 40)).astype(int)
        
        # 5. Kuraklik ve Yayilma Etkilesimleri 
        data['double_drought'] = ((data['FFMC'] >= 88.0) & (data['DC'] >= 500.0)).astype(int)
        data['ISI_x_DC'] = data['ISI'] * data['DC']

        # 6. Yagmur salteri
        data['rain'] = (data['rain'] > 0).astype(int)
        
        # 7. Mekansal (Spatial) Risk Ozellikleri (V2 EKSTRA)
        data['distance_to_center'] = np.sqrt((data['X'] - 5)**2 + (data['Y'] - 5)**2)
        
        def calculate_dynamic_distance(row):
            if row['month'] in ['jun', 'jul', 'aug', 'sep']:
                target_x, target_y = 8, 6
            else:
                target_x, target_y = 6, 5
            return np.sqrt((row['X'] - target_x)**2 + (row['Y'] - target_y)**2)
            
        data['dynamic_seasonal_hotspot_distance'] = data.apply(calculate_dynamic_distance, axis=1)

        # 8. Kolon Siralamasi (Modelin tam olarak bekledigi 22'li sira - HAYAT KURTARAN KISIM)
        expected_cols = [
            'X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain',
            'month_sin', 'month_cos', 'day_sin', 'day_cos', 'is_weekend', 'is_peak_season',
            'distance_to_center', 'dynamic_seasonal_hotspot_distance', 'moderate_wind_danger',
            'hot_and_dry', 'double_drought', 'ISI_x_DC'
        ]
        
        return data[expected_cols]
