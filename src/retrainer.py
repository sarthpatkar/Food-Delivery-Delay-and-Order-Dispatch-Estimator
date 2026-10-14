import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import PRIMARY_DATASET_PATH, RANDOM_STATE, TEST_SIZE
from src.data_pipeline import DeliveryDataPipeline, has_target_column
from src.models import DeliveryModelSuite
from src.evaluate import evaluate_suite, run_cross_validation


class DynamicRetrainingManager:
    def __init__(self):
        self.pipeline = None
        self.model_suite = None
        self.current_metrics = None
        self.cv = None
        self.split = None
        self.cleaned_df = None
        self.total_samples = 0
        self.last_retrain_time = None

    def _fit(self, raw_df):
        """Refit the pipeline, train on 80% and score on the untouched 20%."""
        self.pipeline = DeliveryDataPipeline()
        X, y_reg, y_clf, self.cleaned_df = self.pipeline.fit_transform(raw_df)
        X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
            X, y_reg, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_clf
        )
        self.split = {"X_train": X_train, "y_train_clf": yc_train}

        self.model_suite = DeliveryModelSuite()
        self.model_suite.train(X_train, yr_train, yc_train, self.pipeline.feature_columns)

        self.total_samples = len(X)
        self.current_metrics = evaluate_suite(self.model_suite, X_test, yr_test, yc_test)
        self.last_retrain_time = time.strftime("%Y-%m-%d %H:%M:%S")

    def initialize_baseline(self, dataset_path=PRIMARY_DATASET_PATH):
        self._fit(pd.read_csv(dataset_path))
        tree = DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_split=20, random_state=RANDOM_STATE
        )
        self.cv = run_cross_validation(tree, self.split["X_train"], self.split["y_train_clf"])

    def retrain_with_new_data(self, new_csv_path):
        start_time = time.time()
        new_df = pd.read_csv(new_csv_path)
        if not has_target_column(new_df.columns):
            raise ValueError(
                "The file needs a delivery time column, e.g. 'Time_taken (min)', "
                "'delivery_time_min' or 'actual_delivery_minutes'."
            )

        previous = self.current_metrics
        base_df = pd.read_csv(PRIMARY_DATASET_PATH)
        self._fit(pd.concat([base_df, new_df], ignore_index=True))

        prev_acc = previous["classification"][0]["accuracy"]
        new_acc = self.current_metrics["classification"][0]["accuracy"]
        return {
            "status": "success",
            "rows_added": len(new_df),
            "samples_trained": self.total_samples,
            "training_time_seconds": round(time.time() - start_time, 2),
            "previous_accuracy": prev_acc,
            "updated_accuracy": new_acc,
            "accuracy_delta": round(new_acc - prev_acc, 2),
            "previous_metrics": previous,
            "updated_metrics": self.current_metrics,
            "timestamp": self.last_retrain_time,
        }
