import time
import pandas as pd
from src.config import PRIMARY_DATASET_PATH, SECONDARY_DATASET_PATH
from src.data_pipeline import clean_food_delivery_data, DeliveryDataPipeline
from src.models import DeliveryModelSuite
from src.evaluate import evaluate_suite


class DynamicRetrainingManager:
    def __init__(self):
        self.pipeline = None
        self.model_suite = None
        self.baseline_metrics = None
        self.current_metrics = None
        self.total_samples = 0
        self.last_retrain_time = None

    def initialize_baseline(self, dataset_path=PRIMARY_DATASET_PATH):
        raw_df = pd.read_csv(dataset_path)
        self.total_samples = len(raw_df)

        self.pipeline = DeliveryDataPipeline()
        X, y_reg, y_clf, cleaned_df = self.pipeline.fit_transform(raw_df)

        self.model_suite = DeliveryModelSuite()
        self.model_suite.train(X, y_reg, y_clf, self.pipeline.feature_columns)

        split_idx = int(0.8 * len(X))
        X_eval = X.iloc[split_idx:]
        y_reg_eval = y_reg[split_idx:]
        y_clf_eval = y_clf[split_idx:]

        self.baseline_metrics = evaluate_suite(
            self.model_suite, X_eval, y_reg_eval, y_clf_eval
        )
        self.current_metrics = self.baseline_metrics
        self.last_retrain_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.model_suite.save()

    def retrain_with_new_data(self, new_csv_path, merge_with_base=True):
        start_time = time.time()
        new_df = pd.read_csv(new_csv_path)

        if merge_with_base:
            base_df = pd.read_csv(PRIMARY_DATASET_PATH)
            merged_raw = pd.concat([base_df, new_df], ignore_index=True)
        else:
            merged_raw = new_df

        self.total_samples = len(merged_raw)

        self.pipeline = DeliveryDataPipeline()
        X, y_reg, y_clf, cleaned_df = self.pipeline.fit_transform(merged_raw)

        self.model_suite = DeliveryModelSuite()
        self.model_suite.train(X, y_reg, y_clf, self.pipeline.feature_columns)

        split_idx = int(0.8 * len(X))
        X_eval = X.iloc[split_idx:]
        y_reg_eval = y_reg[split_idx:]
        y_clf_eval = y_clf[split_idx:]

        updated_metrics = evaluate_suite(
            self.model_suite, X_eval, y_reg_eval, y_clf_eval
        )
        elapsed_sec = round(time.time() - start_time, 2)

        prev_acc = self.current_metrics["classification"][0]["accuracy"]
        new_acc = updated_metrics["classification"][0]["accuracy"]

        self.current_metrics = updated_metrics
        self.last_retrain_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.model_suite.save()

        return {
            "status": "success",
            "samples_trained": self.total_samples,
            "training_time_seconds": elapsed_sec,
            "previous_accuracy": prev_acc,
            "updated_accuracy": new_acc,
            "accuracy_delta": round(new_acc - prev_acc, 2),
            "updated_metrics": updated_metrics,
            "timestamp": self.last_retrain_time,
        }
