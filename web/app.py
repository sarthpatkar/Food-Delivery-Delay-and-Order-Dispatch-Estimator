from flask import Flask, render_template, request, jsonify

from src.config import PRIMARY_DATASET_PATH, SECONDARY_DATASET_PATH, FAST_DATASET_PATH, UPLOAD_PATH
from src.clustering import DeliveryClusterAnalyzer
from src.search_planner import UrbanDeliveryGraph
from src.retrainer import DynamicRetrainingManager

app = Flask(__name__)

PRESETS = {"orders": SECONDARY_DATASET_PATH, "times": FAST_DATASET_PATH}
state = {}


def get_state():
    if not state:
        retrainer = DynamicRetrainingManager()
        retrainer.initialize_baseline(PRIMARY_DATASET_PATH)

        clusters = DeliveryClusterAnalyzer(n_clusters=3)
        clusters.compute_elbow_curve(retrainer.split["X_train"])
        clusters.fit_clusters_and_pca(retrainer.split["X_train"], retrainer.cleaned_df)

        graph = UrbanDeliveryGraph()
        if graph.load_street_routes():
            print("Street paths loaded from OSRM", flush=True)
        else:
            print("OSRM unreachable, drawing links as straight lines", flush=True)

        state.update(retrainer=retrainer, clusters=clusters, graph=graph)
    return state


@app.route("/")
def index():
    graph = get_state()["graph"]
    places = [{"id": k, **v} for k, v in graph.nodes.items()]
    return render_template("predict.html", places=places)


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

    retrainer = s["retrainer"]
    sample = retrainer.pipeline.transform_single_input(input_dict)
    prediction = retrainer.model_suite.predict_sample(sample)

    graph = s["graph"]
    drops = [d for d in req.get("drops", []) if isinstance(d, str)][:4]
    trip = graph.plan_trip(
        start=req.get("start_node", graph.start_node),
        drops=drops,
        traffic=input_dict["road_traffic_density"],
        risk=prediction["delay_probability_pct"] / 100.0,
    )

    return jsonify({
        "status": "success",
        "prediction": prediction,
        "stops": trip["stops"],
        "search_comparison": trip["results"],
        "map": graph.map_data(),
    })


@app.route("/retrain")
def retrain_page():
    r = get_state()["retrainer"]
    return render_template(
        "retrain.html",
        metrics=r.current_metrics,
        samples=r.total_samples,
        timestamp=r.last_retrain_time,
    )


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    retrainer = get_state()["retrainer"]
    choice = request.form.get("dataset_choice", "orders")
    upload = request.files.get("custom_file")

    if choice == "custom":
        if not upload or not upload.filename:
            return jsonify({"status": "error", "message": "Choose a CSV file to upload."}), 400
        upload.save(UPLOAD_PATH)
        csv_path = UPLOAD_PATH
    else:
        csv_path = PRESETS.get(choice, SECONDARY_DATASET_PATH)

    try:
        return jsonify(retrainer.retrain_with_new_data(csv_path))
    except (ValueError, KeyError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.route("/explain")
def explain_page():
    s = get_state()
    retrainer, clusters = s["retrainer"], s["clusters"]
    cluster_res = {
        "elbow_data": clusters.elbow_data,
        "profiles": clusters.cluster_profiles,
        "explained_variance": clusters.explained_variance,
        "pca_points": clusters.pca_coordinates,
    }
    return render_template(
        "explain.html",
        metrics=retrainer.current_metrics,
        cv=retrainer.cv,
        clusters=cluster_res,
        tree_rules=retrainer.model_suite.get_decision_tree_rules(max_depth=3),
        feature_importances=retrainer.model_suite.get_feature_importances(),
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
