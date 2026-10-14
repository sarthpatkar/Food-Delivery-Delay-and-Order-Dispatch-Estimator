# Food Delivery Delay & Dispatch Estimator

An AI/ML lab project on the 45,584-row Kaggle food delivery dataset. It predicts delivery time and delay risk, groups orders into clusters, and plans multi-stop delivery trips across Mumbai with A* search, all in a small Flask web app.

---

## Quick start

### 1. What you need

- **Python 3.10 or newer.** Check with `python3 --version` (on Windows, `py --version`).
- **Git**
- **An internet connection** while using the app. The charts, map tiles and street routes load from public servers. No accounts or API keys are needed.

### 2. Get the code

```bash
git clone https://github.com/sarthpatkar/Food-Delivery-Delay-and-Order-Dispatch-Estimator.git
cd Food-Delivery-Delay-and-Order-Dispatch-Estimator
```

The datasets are in the repo under `dataset/`, so there's nothing extra to download.

### 3. Create a virtual environment and install packages

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activate script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again, or use Command Prompt with `.venv\Scripts\activate.bat`.

### 4. Run it

```bash
python run.py
```

Open **http://127.0.0.1:5000** in Chrome, Edge, Firefox or Safari. The first page load takes about 5 to 10 seconds while the models train. After that, everything responds immediately.

Stop the server with `Ctrl+C`. Next time, just activate the virtual environment (step 3, second line) and run `python run.py` again.

### Troubleshooting

| Problem | Fix |
|---|---|
| `Address already in use` / port 5000 busy (common on macOS, where AirPlay Receiver uses it) | Run on another port: `PORT=5050 python run.py` (Windows PowerShell: `$env:PORT=5050; python run.py`), then open http://127.0.0.1:5050 |
| `ModuleNotFoundError` | The virtual environment isn't active. Activate it and run again. |
| `python` not found | Use `python3` on macOS / Linux or `py` on Windows. |
| Map or charts are blank | Check your internet connection. The map, tiles and chart library load from CDNs. |
| Route lines are straight instead of following streets | The public OSRM router was unreachable at startup; the console shows `OSRM unreachable`. Restart the server when you're online. |

---

## Using the app

- **Predict (`/`)**
  - Set the order details. The predicted delivery time, delay probability and four classifier votes update live.
  - Under *Trip in Mumbai*, pick a pickup point and one drop-off per order on the trip. *Other orders on the trip* controls how many drop-off fields appear.
  - The distance input fills in automatically with the straight-line pickup to drop-off 1 distance, which is how the training data measures it. You can drag it to override.
  - The map shows the A* trip on real streets, the Greedy path for comparison, and how busy each road is.
- **Retrain (`/retrain`)**
  - Add `food_delivery_orders.csv`, `food_delivery_times.csv` or your own CSV to the training data and refit every model.
  - Scores before and after are shown side by side.
  - A custom CSV needs a delivery time column, such as `Time_taken (min)`, `delivery_time_min` or `actual_delivery_minutes`.
- **Models (`/explain`)**: confusion matrix, ROC curve, 5-fold cross-validation, K-Means elbow, PCA clusters, feature importance, decision tree rules and the A* cost formula.

---

## What's inside (syllabus mapping)

- **Module 2, search**
  - A* over 12 real Mumbai locations and 18 links. Each link's street path and driving length come from the public OSRM router at startup.
  - The heuristic is the Haversine straight-line distance, which never overestimates.
  - Road cost grows with traffic and with the predicted delay risk.
  - With several drop-offs, every visiting order is costed with A* and the cheapest is used.
  - Greedy best-first and breadth-first search are baselines, costed the same way.
- **Module 3, supervised learning**: multiple linear regression for delivery minutes; decision trees (Gini and entropy), random forest and logistic regression for delay risk (late = over 30 minutes).
- **Module 4, unsupervised learning**: K-Means with an elbow curve, and PCA to plot the clusters in 2D.
- **Module 5, evaluation**: confusion matrix, precision, recall, F1, ROC-AUC and 5-fold stratified cross-validation, all on a held-out 20% test split.
- **Module 6, application**: Flask app with live prediction, retraining without a restart, and Plotly charts.

---

## Project layout

```
.
├── dataset/
│   ├── Dataset.csv                  # Kaggle data, 45,584 orders with GPS coordinates
│   ├── food_delivery_times.csv      # 6k orders, optional retraining batch
│   └── food_delivery_orders.csv     # 1.8k orders with a different schema, optional retraining batch
├── src/
│   ├── config.py                    # Paths and constants
│   ├── data_pipeline.py             # Cleaning, Haversine distance, imputation, scaling
│   ├── models.py                    # Regression and classifiers
│   ├── clustering.py                # K-Means and PCA
│   ├── search_planner.py            # Mumbai network, OSRM street paths, A*, Greedy, BFS
│   ├── evaluate.py                  # Metrics and cross-validation
│   └── retrainer.py                 # Train/test split and retraining
├── web/
│   ├── app.py                       # Flask routes
│   ├── templates/                   # Predict, Retrain and Models pages
│   └── static/                      # CSS and JavaScript (Plotly charts, MapLibre map)
├── reports/project_report.md        # Report and viva questions
├── requirements.txt
└── run.py                           # Start the app
```

---

## Results

Held-out 20% test split (9,117 orders); an order counts as delayed when it takes more than 30 minutes.

- **Random forest**: accuracy 90.87%, ROC-AUC 0.966
- **Decision tree (Gini)**: accuracy 90.36%, ROC-AUC 0.934
- **Linear regression**: MAE 4.93 min, RMSE 6.17 min, R squared 0.567
- **5-fold CV (Gini tree, training split)**: 89.43% +/- 0.44
- **A\* from Lower Parel to Powai, medium traffic**: 30.0 km and 85.8 min via Worli and the sea link. Greedy and BFS go through Dadar and Kurla instead: 27.6 km and 102.6 min.

---

## Credits

Map tiles: CARTO. Map data and routing: OpenStreetMap contributors, via OSRM. Map rendering: MapLibre GL. Charts: Plotly.
