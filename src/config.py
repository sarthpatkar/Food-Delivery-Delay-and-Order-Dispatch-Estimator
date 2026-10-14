from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "dataset"

PRIMARY_DATASET_PATH = DATA_DIR / "Dataset.csv"
FAST_DATASET_PATH = DATA_DIR / "food_delivery_times.csv"
SECONDARY_DATASET_PATH = DATA_DIR / "food_delivery_orders.csv"
UPLOAD_PATH = DATA_DIR / "temp_uploaded.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
DELAY_THRESHOLD_MINUTES = 30

NUMERICAL_FEATURES = [
    "delivery_person_age",
    "delivery_person_ratings",
    "distance_km",
    "vehicle_condition",
    "multiple_deliveries",
]

CATEGORICAL_FEATURES = [
    "weather_conditions",
    "road_traffic_density",
    "type_of_vehicle",
    "city",
]

TARGET_REGRESSION = "time_taken_min"
TARGET_CLASSIFICATION = "is_delayed"
