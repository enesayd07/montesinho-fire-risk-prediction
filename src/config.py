import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TWEEDIE_P = 1.543
TWEEDIE_POWER = 2 - TWEEDIE_P  # 0.457

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "forestfires.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_forestfires_v2.csv")
COLLECTED_DATA_PATH = os.path.join(DATA_DIR, "collected", "new_data.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
CHAMPION_MODEL_PATH = os.path.join(MODEL_DIR, "sklearn_node.pkl")

TARGET_COL = "area"
RAW_FEATURES = ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"]

CATEGORICAL_COLS = ["is_weekend", "is_peak_season", "hot_and_dry", "moderate_wind_danger", "double_drought", "rain"]
CONTINUOUS_COLS = [
    "X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind",
    "month_sin", "month_cos", "day_sin", "day_cos", "ISI_x_DC",
    "distance_to_center", "dynamic_seasonal_hotspot_distance"
]