import os
from flask import Flask, render_template, request, jsonify
import pandas as pd

from src.config import PRIMARY_DATASET_PATH, SECONDARY_DATASET_PATH
from src.data_pipeline import DeliveryDataPipeline, load_and_prepare_data
from src.models import DeliveryModelSuite
from src.clustering import DeliveryClusterAnalyzer
from src.search_planner import UrbanDeliveryGraph
from src.evaluate import evaluate_suite, run_cross_validation
from src.retrainer import DynamicRetrainingManager

app = Flask(__name__)

# Global runtime state
state = {
    "retrainer": None,
    "cluster_analyzer": None,
    "graph": None,
    "data": None,
}


def get_state():
    if state["retrainer"] is None:
        retrainer = DynamicRetrainingManager()
        retrainer.initialize_baseline(PRIMARY_DATASET_PATH)

        data = load_and_prepare_data(PRIMARY_DATASET_PATH)

        cluster_analyzer = DeliveryClusterAnalyzer(n_clusters=3)
        cluster_analyzer.compute_elbow_curve(data["X_train"])
        cluster_analyzer.fit_clusters_and_pca(data["X_train"], data["cleaned_df"])

        graph = UrbanDeliveryGraph()

        state["retrainer"] = retrainer
        state["cluster_analyzer"] = cluster_analyzer
        state["graph"] = graph
        state["data"] = data

    return state


@app.route("/")
def index():
    s = get_state()
    destinations = [
        {"id": k, "name": v["name"]}
        for k, v in s["graph"].nodes.items()
        if k != "Central_Hub"
    ]
    return render_template("predict.html", destinations=destinations)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    s = get_state()
    req = request.get_json() or {}

    input_dict = {
        "distance_km": float(req.get("distance_km", 5.0)),
        "delivery_person_age": float(req.get("delivery_person_age", 28.0)),
        "delivery_person_ratings": float(req.get("delivery_person_ratings", 4.6)),
        "vehicle_condition": float(req.get("vehicle_condition", 1.0)),
        "multiple_deliveries": float(req.get("multiple_deliveries", 1.0)),
        "weather_conditions": str(req.get("weather_conditions", "Sunny")),
        "road_traffic_density": str(req.get("road_traffic_density", "Medium")),
        "type_of_vehicle": str(req.get("type_of_vehicle", "motorcycle")),
        "city": str(req.get("city", "Metropolitian")),
    }

    pipeline = s["retrainer"].pipeline
    model_suite = s["retrainer"].model_suite

    transformed_sample = pipeline.transform_single_input(input_dict)
    prediction = model_suite.predict_sample(transformed_sample)

    destination_node = req.get("destination_node", "Hinjewadi_Phase1")
    search_comparison = s["graph"].compare_all(
        start_node="Central_Hub",
        goal_node=destination_node,
        traffic_override=input_dict["road_traffic_density"],
    )

    return jsonify({
        "status": "success",
        "inputs": input_dict,
        "prediction": prediction,
        "search_comparison": search_comparison,
    })


@app.route("/retrain")
def retrain_page():
    s = get_state()
    metrics = s["retrainer"].current_metrics
    samples = s["retrainer"].total_samples
    timestamp = s["retrainer"].last_retrain_time
    return render_template(
        "retrain.html",
        metrics=metrics,
        samples=samples,
        timestamp=timestamp,
    )


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    s = get_state()
    target_dataset = request.form.get("dataset_choice", "secondary")

    if "custom_file" in request.files and request.files["custom_file"].filename != "":
        uploaded_file = request.files["custom_file"]
        temp_path = "dataset/temp_uploaded.csv"
        uploaded_file.save(temp_path)
        csv_to_use = temp_path
    elif target_dataset == "secondary":
        csv_to_use = str(SECONDARY_DATASET_PATH)
    else:
        csv_to_use = str(PRIMARY_DATASET_PATH)

    result = s["retrainer"].retrain_with_new_data(csv_to_use, merge_with_base=True)
    return jsonify(result)


@app.route("/explain")
def explain_page():
    s = get_state()
    metrics = s["retrainer"].current_metrics
    cluster_res = {
        "elbow_data": s["cluster_analyzer"].elbow_data,
        "profiles": s["cluster_analyzer"].cluster_profiles,
        "explained_variance": s["cluster_analyzer"].explained_variance,
        "pca_points": s["cluster_analyzer"].pca_coordinates,
    }
    tree_rules = s["retrainer"].model_suite.get_decision_tree_rules(max_depth=3)
    feature_importances = s["retrainer"].model_suite.get_feature_importances()

    return render_template(
        "explain.html",
        metrics=metrics,
        clusters=cluster_res,
        tree_rules=tree_rules,
        feature_importances=feature_importances,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
