"""
Pipeline Birim Testleri
12 ham girdi -> 22 öznitelik dönüşümünü doğrular.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from pipeline import FireFeatureEngineeringPipeline


def get_sample_input():

    return pd.DataFrame([{
        "X": 7, "Y": 5, "month": "aug", "day": "fri",
        "FFMC": 96.1, "DMC": 181.1, "DC": 671.2, "ISI": 14.3,
        "temp": 28.7, "RH": 30, "wind": 4.5, "rain": 0.0
    }])


class TestPipelineOutputShape:
    def test_output_has_22_columns(self):
        pipe = FireFeatureEngineeringPipeline()
        result = pipe.transform(get_sample_input())
        assert result.shape[1] == 22, f"Beklenen 22 sütun, gelen {result.shape[1]}"

    def test_month_and_day_dropped(self):
        pipe = FireFeatureEngineeringPipeline()
        result = pipe.transform(get_sample_input())
        assert "month" not in result.columns, "month sütunu düşürülmemiş"
        assert "day" not in result.columns, "day sütunu düşürülmemiş"
        assert "month_num" not in result.columns, "month_num geçici sütunu düşürülmemiş"
        assert "day_num" not in result.columns, "day_num geçici sütunu düşürülmemiş"


class TestCyclicEncoding:
    def test_sin_cos_in_range(self):
        pipe = FireFeatureEngineeringPipeline()
        result = pipe.transform(get_sample_input())
        for col in ["month_sin", "month_cos", "day_sin", "day_cos"]:
            val = result[col].iloc[0]
            assert -1.0 <= val <= 1.0, f"{col} değeri {val}, [-1, 1] aralığında değil"


class TestBinaryFeatures:
    def test_rain_zero_becomes_zero(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["rain"] = 0.0
        result = pipe.transform(df)
        assert result["rain"].iloc[0] == 0, "rain=0.0 olduğunda çıktı 0 olmalı"

    def test_rain_positive_becomes_one(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["rain"] = 2.5
        result = pipe.transform(df)
        assert result["rain"].iloc[0] == 1, "rain>0 olduğunda çıktı 1 olmalı"

    def test_weekend_friday_is_zero(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["day"] = "fri"
        result = pipe.transform(df)
        assert result["is_weekend"].iloc[0] == 0, "Cuma hafta sonu değil, 0 olmalı"

    def test_weekend_saturday_is_one(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["day"] = "sat"
        result = pipe.transform(df)
        assert result["is_weekend"].iloc[0] == 1, "Cumartesi hafta sonu, 1 olmalı"

    def test_peak_season_august(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["month"] = "aug"
        result = pipe.transform(df)
        assert result["is_peak_season"].iloc[0] == 1, "Ağustos pik sezon, 1 olmalı"

    def test_not_peak_season_january(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["month"] = "jan"
        result = pipe.transform(df)
        assert result["is_peak_season"].iloc[0] == 0, "Ocak pik sezon değil, 0 olmalı"


class TestInteractionFeatures:
    def test_isi_x_dc_calculation(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        result = pipe.transform(df)
        expected = df["ISI"].iloc[0] * df["DC"].iloc[0]
        assert abs(result["ISI_x_DC"].iloc[0] - expected) < 0.01, "ISI_x_DC hesaplaması hatalı"

    def test_double_drought_active(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["DMC"] = 150.0
        df["DC"] = 600.0
        result = pipe.transform(df)
        assert result["double_drought"].iloc[0] == 1, "DMC>100 ve DC>500 iken double_drought=1 olmalı"

    def test_double_drought_inactive(self):
        pipe = FireFeatureEngineeringPipeline()
        df = get_sample_input()
        df["DMC"] = 50.0
        df["DC"] = 200.0
        result = pipe.transform(df)
        assert result["double_drought"].iloc[0] == 0, "DMC<100 ve DC<500 iken double_drought=0 olmalı"


class TestSpatialFeatures:
    def test_distance_to_center_exists(self):
        pipe = FireFeatureEngineeringPipeline()
        result = pipe.transform(get_sample_input())
        assert "distance_to_center" in result.columns, "distance_to_center sütunu eksik"

    def test_dynamic_seasonal_hotspot_exists(self):
        pipe = FireFeatureEngineeringPipeline()
        result = pipe.transform(get_sample_input())
        assert "dynamic_seasonal_hotspot_distance" in result.columns, "dynamic_seasonal_hotspot_distance sütunu eksik"