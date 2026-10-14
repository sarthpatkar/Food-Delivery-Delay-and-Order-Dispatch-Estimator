import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    PRIMARY_DATASET_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
)


def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    lat1_rad = np.radians(np.abs(lat1))
    lon1_rad = np.radians(np.abs(lon1))
    lat2_rad = np.radians(np.abs(lat2))
    lon2_rad = np.radians(np.abs(lon2))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius_km * c


def clean_food_delivery_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    rename_map = {
        "Delivery_person_Age": "delivery_person_age",
        "Delivery_person_Ratings": "delivery_person_ratings",
        "Weather_conditions": "weather_conditions",
        "Weatherconditions": "weather_conditions",
        "Road_traffic_density": "road_traffic_density",
        "Vehicle_condition": "vehicle_condition",
        "Type_of_order": "type_of_order",
        "Type_of_vehicle": "type_of_vehicle",
        "multiple_deliveries": "multiple_deliveries",
        "Festival": "festival",
        "City": "city",
        "Time_taken (min)": "time_taken_min",
        "Time_taken(min)": "time_taken_min",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "distance_km" not in df.columns:
        if (
            "Restaurant_latitude" in df.columns
            and "Delivery_location_latitude" in df.columns
        ):
            distances = calculate_haversine_distance(
                df["Restaurant_latitude"],
                df["Restaurant_longitude"],
                df["Delivery_location_latitude"],
                df["Delivery_location_longitude"],
            )
            median_dist = distances[(distances > 0.1) & (distances < 50)].median()
            distances = distances.where(
                (distances > 0.1) & (distances < 50), median_dist
            )
            df["distance_km"] = distances
        else:
            df["distance_km"] = 5.0

    if df["time_taken_min"].dtype == object:
        df["time_taken_min"] = (
            df["time_taken_min"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        )

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.strip().str.replace("conditions ", "")
            )
            mode_val = df[col].mode()[0] if not df[col].empty else "Unknown"
            df[col] = df[col].replace({"nan": mode_val, "NaN": mode_val, "": mode_val})

    df[TARGET_CLASSIFICATION] = (df[TARGET_REGRESSION] > 30).astype(int)

    return df


class DeliveryDataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.categorical_columns_encoded = None
        self.medians = {}
        self.modes = {}

    def fit_transform(self, df: pd.DataFrame):
        cleaned_df = clean_food_delivery_data(df)

        for col in NUMERICAL_FEATURES:
            self.medians[col] = float(cleaned_df[col].median())
        for col in CATEGORICAL_FEATURES:
            self.modes[col] = str(cleaned_df[col].mode()[0])

        num_data = cleaned_df[NUMERICAL_FEATURES]
        cat_data = pd.get_dummies(
            cleaned_df[CATEGORICAL_FEATURES], drop_first=True, dtype=float
        )
        self.categorical_columns_encoded = list(cat_data.columns)

        scaled_num = self.scaler.fit_transform(num_data)
        scaled_num_df = pd.DataFrame(
            scaled_num, columns=NUMERICAL_FEATURES, index=cleaned_df.index
        )

        X = pd.concat([scaled_num_df, cat_data], axis=1)
        self.feature_columns = list(X.columns)

        y_reg = cleaned_df[TARGET_REGRESSION].values
        y_clf = cleaned_df[TARGET_CLASSIFICATION].values

        return X, y_reg, y_clf, cleaned_df

    def transform_single_input(self, input_dict: dict) -> pd.DataFrame:
        raw_row = {}
        for col in NUMERICAL_FEATURES:
            raw_row[col] = float(input_dict.get(col, self.medians.get(col, 0.0)))
        for col in CATEGORICAL_FEATURES:
            raw_row[col] = str(input_dict.get(col, self.modes.get(col, "Unknown")))

        row_df = pd.DataFrame([raw_row])

        num_scaled = self.scaler.transform(row_df[NUMERICAL_FEATURES])
        num_scaled_df = pd.DataFrame(num_scaled, columns=NUMERICAL_FEATURES)

        cat_encoded = pd.get_dummies(
            row_df[CATEGORICAL_FEATURES], drop_first=False, dtype=float
        )
        cat_aligned = pd.DataFrame(
            0.0, index=[0], columns=self.categorical_columns_encoded
        )

        for col in cat_encoded.columns:
            if col in cat_aligned.columns:
                cat_aligned[col] = cat_encoded[col]

        full_sample = pd.concat([num_scaled_df, cat_aligned], axis=1)
        return full_sample[self.feature_columns]


def load_and_prepare_data(csv_path=PRIMARY_DATASET_PATH):
    raw_df = pd.read_csv(csv_path)
    pipeline = DeliveryDataPipeline()
    X, y_reg, y_clf, cleaned_df = pipeline.fit_transform(raw_df)

    (
        X_train,
        X_test,
        y_train_reg,
        y_test_reg,
        y_train_clf,
        y_test_clf,
    ) = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_clf,
    )

    return {
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_reg": y_train_reg,
        "y_test_reg": y_test_reg,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "feature_names": pipeline.feature_columns,
        "cleaned_df": cleaned_df,
    }
