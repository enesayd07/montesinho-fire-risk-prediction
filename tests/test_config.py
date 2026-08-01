"""
Config Tutarlılık Testleri
Sabitlerin ve yol tanımlarının geçerli olduğunu doğrular.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import (
    TWEEDIE_P, TWEEDIE_POWER,
    CATEGORICAL_COLS, CONTINUOUS_COLS, TARGET_COL,
    CHAMPION_MODEL_PATH, RAW_FEATURES
)


class TestTweedieConstants:
    """Tweedie dağılım sabitlerinin geçerliliğini test eder."""

    def test_tweedie_p_in_valid_range(self):
        assert 1.0 < TWEEDIE_P < 2.0, (
            f"TWEEDIE_P={TWEEDIE_P}, (1, 2) aralığında olmalı "
            f"(1=Poisson, 2=Gamma, arası=Compound Poisson-Gamma)"
        )

    def test_tweedie_power_consistency(self):
        expected = 2 - TWEEDIE_P
        assert abs(TWEEDIE_POWER - expected) < 1e-10, (
            f"TWEEDIE_POWER={TWEEDIE_POWER}, 2-TWEEDIE_P={expected} ile tutarsız"
        )


class TestColumnDefinitions:
    """Sütun tanımlarının boş olmadığını ve tutarlı olduğunu test eder."""

    def test_categorical_cols_not_empty(self):
        assert len(CATEGORICAL_COLS) > 0, "CATEGORICAL_COLS boş olmamalı"

    def test_continuous_cols_not_empty(self):
        assert len(CONTINUOUS_COLS) > 0, "CONTINUOUS_COLS boş olmamalı"

    def test_no_overlap_between_cat_and_cont(self):
        overlap = set(CATEGORICAL_COLS) & set(CONTINUOUS_COLS)
        assert len(overlap) == 0, (
            f"Kategorik ve sürekli sütunlar çakışıyor: {overlap}"
        )

    def test_target_col_is_area(self):
        assert TARGET_COL == "area", f"TARGET_COL={TARGET_COL}, 'area' olmalı"

    def test_raw_features_has_12_items(self):
        assert len(RAW_FEATURES) == 12, (
            f"RAW_FEATURES {len(RAW_FEATURES)} eleman içeriyor, 12 olmalı"
        )


class TestPaths:
    """Yol sabitlerinin doğru formatta olduğunu test eder."""

    def test_champion_path_ends_correctly(self):
        assert CHAMPION_MODEL_PATH.endswith("champion_node"), (
            f"CHAMPION_MODEL_PATH '{CHAMPION_MODEL_PATH}' ile bitiyor, "
            f"'champion_node' ile bitmeli"
        )