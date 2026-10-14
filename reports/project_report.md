# AI/ML Lab Mini Project Report

## Project Title: LogiPulse AI - Food Delivery Delay & Order Dispatch Estimator

**Course**: Artificial Intelligence and Machine Learning (AI/ML) Laboratory  
**Dataset**: Official 45,584-row Kaggle Food Delivery Benchmark Dataset (`Dataset.csv`)  
**Technology Stack**: Python 3.13, Scikit-Learn, Pandas, NumPy, SciPy, Flask, Plotly  

---

## 1. Abstract & Problem Statement

Urban on-demand food delivery platforms (such as Swiggy and Zomato) operate under strict Service Level Agreements (SLAs). Inaccurate delivery time estimates and unoptimized dispatch sequences lead to customer dissatisfaction, driver idle times, and operational bottlenecks. 

This project implements an end-to-end AI/ML solution that solves two fundamental challenges:
1. **Accurate Delivery Duration & Delay Risk Prediction**: Multiple Linear Regression forecasts continuous delivery duration in minutes, while Decision Trees (Gini and Entropy) and Random Forest classify whether an order faces severe delay risk.
2. **Dynamic Route Dispatch Optimization**: An informed heuristic search algorithm ($A^*$ Search) optimizes multi-stop delivery sequences by combining geographic Haversine distance with live machine-learning traffic delay penalties.
3. **Unsupervised Zone Discovery**: K-Means clustering with an automated Elbow Method and Principal Component Analysis (PCA) stratifies urban delivery zones into distinct transit speed profiles.
4. **Dynamic Model Retraining**: An interactive studio allows the model suite to re-fit scalers and retrain on new incoming delivery batches dynamically in memory.

---

## 2. Mapping to Course Syllabus & Learning Outcomes

| Syllabus Module & Practical | Topic Covered | Implementation in LogiPulse AI |
|---|---|---|
| **Module 2: Search Strategies** | Informed ($A^*$) and Uninformed (BFS, Greedy) Search | `src/search_planner.py`: Compares $A^*$, Greedy Best-First, and BFS across transit waypoints. |
| **Module 2: Heuristic Design** | Admissible Heuristic Formulation | Great-circle Haversine distance + ML traffic delay weight penalty. |
| **Module 3: Supervised Regression** | Multiple Linear Regression | `src/models.py`: Ordinary Least Squares predicting delivery duration. |
| **Module 3: Supervised Classification** | Decision Trees, Random Forest, Logistic Regression | `src/models.py`: Binary delay risk classification ($1$ vs $0$). |
| **Module 3: Split Criteria Comparison** | Gini Impurity vs Entropy | Evaluates both criteria on decision tree splits with exact threshold extraction. |
| **Module 4: Unsupervised Clustering** | K-Means Clustering & Elbow Method | `src/clustering.py`: Computes WCSS across $k \in [2, 8]$; justifies optimal $k=3$. |
| **Module 4: Dimensionality Reduction** | Principal Component Analysis (PCA) | `src/clustering.py`: Reduces 18 features to 2 principal components; computes explained variance. |
| **Module 5: Model Evaluation** | Confusion Matrix, ROC-AUC, 5-Fold CV | `src/evaluate.py`: Generates TP/FP/TN/FN matrix, ROC curve, and 5-fold cross-validation. |
| **Module 6: Real-World Application** | Dynamic Retraining & Explainability | `web/app.py`: Flask + Plotly dashboard with live prediction and algorithm mechanics. |

---

## 3. Mathematical Foundations

### 3.1 Haversine Distance Formula
Given restaurant coordinates $(\phi_1, \lambda_1)$ and customer drop coordinates $(\phi_2, \lambda_2)$, the great-circle distance $d$ over a spherical Earth radius $R = 6371\text{ km}$ is:

$$\Delta\phi = \phi_2 - \phi_1, \quad \Delta\lambda = \lambda_2 - \lambda_1$$

$$a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)$$

$$d = 2R \cdot \arctan2\left(\sqrt{a}, \sqrt{1-a}\right)$$

### 3.2 A* Heuristic Search Evaluation
The total estimated path cost through node $n$ is defined as:

$$f(n) = g(n) + h(n)$$

Where:
- $g(n)$: Exact accumulated travel cost from start node to current node $n$.
- $h(n)$: Straight-line Haversine distance from node $n$ to goal node. Because Euclidean/Haversine distance is the shortest possible path between two points on Earth, $h(n) \le h^*(n)$ (admissible), guaranteeing that $A^*$ will find the optimal path.
- Edge Weight Function: $w(u, v) = \text{distance}(u, v) \times \text{TrafficFactor}(u, v)$, where TrafficFactor is dynamically weighted by the ML classification layer.

### 3.3 Decision Tree Splitting Criteria
1. **Gini Impurity**:
   $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$
2. **Entropy (Information Gain)**:
   $$H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

### 3.4 K-Means Within-Cluster Sum of Squares (Inertia)
The objective function minimized during Lloyd's iterative algorithm is:

$$\text{WCSS} = \sum_{j=1}^{k} \sum_{x_i \in S_j} ||x_i - \mu_j||^2$$

Where $\mu_j$ represents the centroid of cluster $S_j$.

### 3.5 Classification Evaluation Metrics
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1\text{-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 4. Experimental Results & Performance Analysis

### 4.1 Supervised Classification Benchmark (Held-out Test Set: 9,117 Orders)

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest Classifier** | **90.63%** | **99.42%** | **69.01%** | **81.47%** | **0.965** |
| **Decision Tree (Gini)** | 90.37% | 90.82% | 75.33% | 82.36% | 0.936 |
| **Decision Tree (Entropy)** | 90.30% | 90.80% | 75.11% | 82.21% | 0.937 |
| **Logistic Regression** | 87.84% | 82.57% | 75.07% | 78.64% | 0.939 |

### 4.2 Supervised Regression Performance
- **Mean Absolute Error (MAE)**: 4.93 minutes
- **Root Mean Squared Error (RMSE)**: 6.17 minutes
- **R-Squared ($R^2$) Score**: 0.567

### 4.3 5-Fold Stratified Cross-Validation
- Fold Accuracies: [92.5%, 89.5%, 91.5%, 90.5%, 90.75%]
- **Mean Accuracy**: **90.95%**
- **Standard Deviation**: **1.00%** (indicates outstanding generalization stability without overfitting).

### 4.4 Search Algorithm Performance Comparison (Central Hub -> Hinjewadi Phase 1)

| Algorithm | Strategy | Nodes Expanded | Total Distance (km) | Estimated Transit (min) | Optimality |
|---|---|---|---|---|---|
| **A* Search** | Informed ($g + h + \text{penalty}$) | **9** | 18.6 km | **46.0 min** | **Optimal** (smart detour via Wakad) |
| **Greedy Best-First** | Informed ($h$ only) | 5 | 17.0 km | 49.6 min | Suboptimal (enters traffic bottleneck) |
| **Breadth First (BFS)** | Uninformed (hops) | 12 | 21.0 km | 71.2 min | Poor (meandering path) |

---

## 5. Top 15 Viva Voce Questions & Answers

1. **Why did you use the Haversine formula instead of Euclidean distance?**  
   *Answer*: Earth is an oblate spheroid. Euclidean distance treats coordinates as a flat 2D plane ($x, y$), which distorts true geographic distances as latitude changes. The Haversine formula calculates the true great-circle distance over a sphere.

2. **How does A* search guarantee finding the shortest path?**  
   *Answer*: A* is guaranteed to be optimal if its heuristic function $h(n)$ is admissible (never overestimates the true cost to reach the goal). Because the straight-line Haversine distance is the shortest distance between two points on Earth, it is strictly admissible.

3. **What is the difference between Gini Impurity and Entropy in Decision Trees?**  
   *Answer*: Gini measures the probability of misclassifying a randomly chosen element ($1 - \sum p_i^2$), while Entropy measures information disorder ($-\sum p_i \log_2 p_i$). Entropy involves logarithmic calculations and is computationally more expensive, whereas Gini is faster and slightly favors larger class partitions.

4. **Why did Random Forest outperform a single Decision Tree?**  
   *Answer*: A single Decision Tree is prone to high variance and overfitting on noisy data. Random Forest aggregates multiple decorrelated decision trees trained on bootstrap samples (bagging) with random feature subsets, reducing overall model variance without increasing bias.

5. **How did you handle missing values in the Kaggle dataset?**  
   *Answer*: Numerical features (`Delivery_person_Age`, `Ratings`) were imputed with their median values to maintain robustness against outliers. Categorical features (`Weather_conditions`, `Road_traffic_density`) were imputed using the statistical mode.

6. **What is the purpose of the Elbow Method in K-Means clustering?**  
   *Answer*: K-Means requires the user to specify $k$ beforehand. The Elbow Method plots Within-Cluster Sum of Squares (Inertia) against increasing values of $k$. The inflection point where the rate of inertia reduction sharply decreases indicates the mathematically optimal balance between cluster compactness and model complexity.

7. **Why was PCA applied before cluster visualization?**  
   *Answer*: The preprocessed dataset has 18 features (high dimensional space), making direct visual inspection impossible. PCA projects the feature space onto its top 2 orthogonal eigenvectors (principal components) that capture the maximum variance, allowing 2D cluster visualization.

8. **What does the Area Under the ROC Curve (AUC) tell you?**  
   *Answer*: AUC measures the model's ability to discriminate between positive (delayed) and negative (on-time) classes across all classification probability thresholds. An AUC of 0.965 indicates a 96.5% chance that the model ranks a randomly chosen delayed order higher than an on-time order.

9. **Why is Precision critical in delivery delay risk management?**  
   *Answer*: False Positives occur when the system predicts an order will be delayed when it will actually arrive on time. High precision (99.42% in our Random Forest) ensures the platform does not unnecessarily compensate customers or send premature delay panic alerts.

10. **How does the Dynamic Retraining feature work in your project?**  
    *Answer*: The system accepts new delivery records or uploaded CSVs, concatenates them with historical training data, re-fits the `StandardScaler` and one-hot encoders, and retrains the estimators in memory (~2 seconds) without requiring a server reboot.

11. **What is Stratified K-Fold Cross-Validation?**  
    *Answer*: Unlike standard K-Fold, Stratified K-Fold ensures that each fold contains approximately the same percentage of samples of each target class as the complete dataset. This prevents fold bias, especially when handling imbalanced classes.

12. **Why was k-Nearest Neighbors excluded from the final pipeline?**  
    *Answer*: k-NN is a lazy learner with $O(N)$ inference complexity. For a 45,000-row dataset, calculating Euclidean distances to all points for every live user query creates noticeable latency, whereas Decision Trees and Random Forests execute in $O(\text{depth})$ time.

13. **What is the significance of the R-Squared ($R^2$) metric in your regression model?**  
    *Answer*: $R^2$ represents the proportion of variance in delivery duration that is predictable from our independent features (distance, ratings, weather, traffic). An $R^2$ of 0.567 indicates that over 56% of delivery time variation is directly explained by traffic, distance, and rider factors.

14. **How do traffic penalties alter the graph edge weights in A* search?**  
    *Answer*: The physical road distance $d$ is scaled by a dynamic traffic multiplier: $w = d \times \text{Multiplier}$. Low traffic has multiplier 1.0x, Medium has 1.35x, High has 1.75x, and Jam has 2.2x. This penalizes congested paths and causes $A^*$ to explore faster detours.

15. **Is the trained model suite stored permanently or retrained on every boot?**  
    *Answer*: Trained model suites are serialized to `models_cache/model_suite.joblib`. On application startup, the cached models load instantly. Retraining is only triggered when the user explicitly requests dynamic retraining via the `/retrain` studio.

---

## 6. Conclusion
LogiPulse AI successfully bridges theoretical AI/ML concepts from Modules 0 through 6 into a high-performance, real-world application. By integrating multiple regression, decision tree classification, K-Means clustering, and $A^*$ search with real Kaggle delivery logs, the system demonstrates strong predictive accuracy (90.63% classification accuracy, 0.965 AUC) and provides an explainable, interactive interface for both academic defense and operational deployment.
