# LogiPulse AI: Food Delivery Delay & Dispatch Estimator

An end-to-end laboratory AI/ML project integrating Supervised Regression, Decision Trees, Random Forest, K-Means Clustering, PCA, and A* Route Search on the official 45,584-row Kaggle Food Delivery dataset.

---

## Features & Syllabus Mapping

- **Module 2 (Informed & Uninformed Search)**:
  - $A^*$ Search Algorithm with admissible Haversine heuristic + ML traffic penalty.
  - Greedy Best-First Search and Breadth First Search (BFS) baselines for performance comparison.
- **Module 3 (Supervised Learning)**:
  - Multiple Linear Regression forecasting delivery minutes.
  - Decision Tree Classifier comparing Gini Impurity vs Entropy criteria.
  - Random Forest Classifier (ensemble comparison).
  - Logistic Regression baseline.
- **Module 4 (Unsupervised Learning)**:
  - K-Means Clustering with automated Elbow Method (Inertia curve).
  - Principal Component Analysis (PCA) 2D projection and variance analysis.
- **Module 5 (Model Evaluation & Optimization)**:
  - Confusion Matrix heatmap (TP, FP, TN, FN).
  - Precision, Recall, F1-Score, and ROC Curve with AUC calculation.
  - 5-Fold Stratified Cross-Validation.
- **Module 6 (Application & Dynamic Retraining)**:
  - Flask web application with 3 dedicated interactive views.
  - Dynamic Retraining Studio allowing live re-fitting on new datasets without server reboot.
  - Real-Time Backend & Algorithm Mechanics view with interactive Plotly charts.
- **Code Standards**:
  - Zero emojis in code, logs, and UI.
  - Clean, human-written Python with minimal and natural comments.

---

## Directory Structure

```
aiml/
├── dataset/
│   ├── Dataset.csv                      # Primary 45,584-row Kaggle benchmark with GPS
│   ├── food_delivery_times.csv          # Fast 6k pre-computed feature dataset
│   └── food_delivery_orders.csv         # Secondary dataset for dynamic retraining
├── src/
│   ├── __init__.py
│   ├── config.py                        # Constants and path definitions
│   ├── data_pipeline.py                 # Coordinate cleaning, Haversine, imputation, scaling
│   ├── models.py                        # LinearRegression, DecisionTree, Random Forest
│   ├── clustering.py                    # K-Means (Elbow method), PCA
│   ├── search_planner.py                # A* Search, Greedy, and BFS dispatch planners
│   ├── evaluate.py                      # Confusion matrix, ROC curve, cross-validation
│   └── retrainer.py                     # Dynamic retraining controller
├── web/
│   ├── app.py                           # Flask web server and API endpoints
│   ├── templates/
│   │   ├── base.html                    # Base layout (no emojis)
│   │   ├── predict.html                 # Prediction and A* dispatch portal
│   │   ├── retrain.html                 # Dynamic retraining studio
│   │   └── explain.html                 # Real-time algorithm mechanics & Plotly charts
│   └── static/
│       ├── css/style.css                # Modern CSS stylesheet
│       └── js/main.js                   # Client-side API interactions & Plotly graphs
├── reports/
│   └── project_report.md                # Academic report + Top 15 Viva Q&A
├── models_cache/                        # Serialized trained model suites
├── requirements.txt
└── run.py                               # Application runner script
```

---

## How to Run

1. **Launch the Web Application**:
   ```bash
   python3 run.py
   ```
2. **Open the browser**:
   Navigate to: `http://127.0.0.1:5000`

3. **Navigate the 3 Dedicated Tabs**:
   - `/` — Order Predictor & A* Dispatch: Adjust sliders to test delivery times and route optimizations.
   - `/retrain` — Dynamic Retraining Studio: Retrain models on new cohorts with live metric updates.
   - `/explain` — Real-Time Algorithm Mechanics: Inspect the interactive Confusion Matrix, ROC-AUC, K-Means Elbow curve, and Decision Tree split rules.

---

## Evaluation Benchmark Summary

- **Random Forest Accuracy**: 90.63% | **ROC-AUC**: 0.965
- **Decision Tree Accuracy**: 90.37% | **ROC-AUC**: 0.936
- **Multiple Linear Regression MAE**: 4.93 minutes | **RMSE**: 6.17 minutes
- **5-Fold Cross-Validation**: 90.95% ± 1.0%
- **A* Search**: 18.6 km, 46.0 min (optimal detour avoiding traffic jam)
