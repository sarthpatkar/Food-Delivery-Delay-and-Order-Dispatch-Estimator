# AI/ML Lab Mini Project Report

## Project Title: Food Delivery Delay & Order Dispatch Estimator

**Course**: Artificial Intelligence and Machine Learning (AI/ML) Laboratory  
**Dataset**: Official 45,584-row Kaggle Food Delivery Benchmark Dataset (`Dataset.csv`)  
**Technology Stack**: Python 3.13, Scikit-Learn, Pandas, NumPy, SciPy, Flask, Plotly  

---

## 1. Abstract & Problem Statement

Urban on-demand food delivery platforms (such as Swiggy and Zomato) operate under strict Service Level Agreements (SLAs). Inaccurate delivery time estimates and unoptimized dispatch sequences lead to customer dissatisfaction, driver idle times, and operational bottlenecks. 

This project implements an end-to-end AI/ML solution that solves two fundamental challenges:
1. **Accurate Delivery Duration & Delay Risk Prediction**: Multiple Linear Regression forecasts continuous delivery duration in minutes, while Decision Trees (Gini and Entropy) and Random Forest classify whether an order faces severe delay risk.
2. **Route Planning**: $A^*$ search finds the cheapest route across a sample Mumbai road network. Road cost combines distance, the traffic level and the delay probability predicted by the classifiers.
3. **Unsupervised Zone Discovery**: K-Means clustering with an automated Elbow Method and Principal Component Analysis (PCA) stratifies urban delivery zones into distinct transit speed profiles.
4. **Retraining**: New delivery batches, including files with a different column layout, can be added and every model refitted in memory.

---

## 2. Mapping to Course Syllabus & Learning Outcomes

| Syllabus Module & Practical | Topic Covered | Implementation |
|---|---|---|
| **Module 2: Search Strategies** | Informed ($A^*$) and Uninformed (BFS, Greedy) Search | `src/search_planner.py`: Compares $A^*$, Greedy best-first and BFS on a 12-node Mumbai network whose links follow real streets (OSRM). |
| **Module 2: Heuristic Design** | Admissible Heuristic Formulation | Great-circle Haversine distance; every road is at least that long, so $h$ never overestimates. |
| **Module 3: Supervised Regression** | Multiple Linear Regression | `src/models.py`: Ordinary Least Squares predicting delivery duration. |
| **Module 3: Supervised Classification** | Decision Trees, Random Forest, Logistic Regression | `src/models.py`: Binary delay risk classification ($1$ vs $0$). |
| **Module 3: Split Criteria Comparison** | Gini Impurity vs Entropy | Evaluates both criteria on decision tree splits with exact threshold extraction. |
| **Module 4: Unsupervised Clustering** | K-Means Clustering & Elbow Method | `src/clustering.py`: Computes WCSS across $k \in [2, 8]$; $k=3$ is used. |
| **Module 4: Dimensionality Reduction** | Principal Component Analysis (PCA) | `src/clustering.py`: Reduces 18 features to 2 principal components; computes explained variance. |
| **Module 5: Model Evaluation** | Confusion Matrix, ROC-AUC, 5-Fold CV | `src/evaluate.py`: Generates TP/FP/TN/FN matrix, ROC curve, and 5-fold cross-validation. |
| **Module 6: Real-World Application** | Dynamic Retraining & Explainability | `web/app.py`: Flask app with live prediction, retraining and Plotly charts. |

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
- Edge weight: $w(u, v) = d(u, v) \cdot (m + r(m - 1))$, where $m$ is the traffic multiplier of the road (Low 1.0, Medium 1.35, High 1.75, Jam 2.2) and $r$ is the predicted delay probability. The chosen traffic level shifts every road up or down one level. Since $w \ge d \ge h$, the heuristic stays admissible.

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
| **Random forest** | **90.87%** | **99.68%** | **69.63%** | **81.99%** | **0.966** |
| **Decision tree (Gini)** | 90.36% | 90.82% | 75.29% | 82.33% | 0.934 |
| **Decision tree (entropy)** | 90.36% | 90.82% | 75.29% | 82.33% | 0.934 |
| **Logistic regression** | 87.90% | 82.99% | 74.78% | 78.67% | 0.936 |

Models are trained on the other 80% only. An order counts as delayed when it takes more than 30 minutes (29.8% of orders).

### 4.2 Supervised Regression Performance
- **Mean Absolute Error (MAE)**: 4.93 minutes
- **Root Mean Squared Error (RMSE)**: 6.17 minutes
- **R-Squared ($R^2$) Score**: 0.567

### 4.3 5-Fold Stratified Cross-Validation
Gini decision tree, run on the training split.
- Fold accuracies: 89.70%, 89.06%, 89.11%, 90.17%, 89.10%
- **Mean accuracy**: **89.43%**
- **Standard deviation**: **0.44** points, so the score barely moves between folds.

### 4.4 Search Comparison (Lower Parel to Powai, medium traffic, 7.3% delay risk)

| Algorithm | Strategy | Nodes Expanded | Distance | Est. Time | Path Cost |
|---|---|---|---|---|---|
| **A* search** | $g + h$ | 11 | 30.0 km | **85.8 min** | **43.82** (optimal) |
| **Greedy best-first** | $h$ only | **5** | 27.6 km | 102.6 min | 53.03 |
| **Breadth-first** | fewest hops | 11 | 27.6 km | 102.6 min | 53.03 |

Link lengths are real driving distances on the street network, fetched once at startup from the public OSRM router.

A* takes the longer western route through Worli, the sea link and Santacruz because the Dadar to Kurla road is jammed. Greedy follows the straight line into that jam. Under heavy traffic with a high delay risk (Jam, 88.5%) every road is congested, and all three choose the shorter Dadar, Kurla and Ghatkopar route.

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
   *Answer*: K-Means needs $k$ up front. The elbow method plots WCSS against $k$ and looks for the point where adding clusters stops paying off. On this data the curve has no sharp elbow, so $k = 3$ was picked where the drop eases and because three groups (quick, typical, long runs) are easy to interpret.

7. **Why was PCA applied before cluster visualization?**  
   *Answer*: The preprocessed dataset has 18 features (high dimensional space), making direct visual inspection impossible. PCA projects the feature space onto its top 2 orthogonal eigenvectors (principal components) that capture the maximum variance, allowing 2D cluster visualization.

8. **What does the Area Under the ROC Curve (AUC) tell you?**  
   *Answer*: AUC measures the model's ability to discriminate between positive (delayed) and negative (on-time) classes across all classification probability thresholds. An AUC of 0.966 indicates a 96.6% chance that the model ranks a randomly chosen delayed order higher than an on-time order.

9. **Why is Precision critical in delivery delay risk management?**  
   *Answer*: False Positives occur when the system predicts an order will be delayed when it will actually arrive on time. High precision (99.68% in our random forest) ensures the platform does not unnecessarily compensate customers or send premature delay panic alerts.

10. **How does the Dynamic Retraining feature work in your project?**  
    *Answer*: The system accepts new delivery records or uploaded CSVs, concatenates them with historical training data, re-fits the `StandardScaler` and one-hot encoders, and retrains the estimators in memory (under a second on this machine) without restarting the server. Column names from the other two datasets, such as `actual_delivery_minutes` or `courier_distance_km`, are mapped to the Kaggle names first, and a file with no delivery time column is rejected with a clear message.

11. **What is Stratified K-Fold Cross-Validation?**  
    *Answer*: Unlike standard K-Fold, Stratified K-Fold ensures that each fold contains approximately the same percentage of samples of each target class as the complete dataset. This prevents fold bias, especially when handling imbalanced classes.

12. **Why was k-Nearest Neighbors excluded from the final pipeline?**  
    *Answer*: k-NN is a lazy learner with $O(N)$ inference complexity. For a 45,000-row dataset, calculating Euclidean distances to all points for every live user query creates noticeable latency, whereas Decision Trees and Random Forests execute in $O(\text{depth})$ time.

13. **What is the significance of the R-Squared ($R^2$) metric in your regression model?**  
    *Answer*: $R^2$ represents the proportion of variance in delivery duration that is predictable from our independent features (distance, ratings, weather, traffic). An $R^2$ of 0.567 indicates that over 56% of delivery time variation is directly explained by traffic, distance, and rider factors.

14. **How do traffic penalties alter the graph edge weights in A* search?**  
    *Answer*: Each road has a usual traffic level. The traffic chosen in the form shifts every road up or down one level, and the cost is $w = d(m + r(m - 1))$ with multiplier $m$ (Low 1.0, Medium 1.35, High 1.75, Jam 2.2) and predicted delay probability $r$. A risky order therefore pays extra for congested roads, which can push $A^*$ onto a longer but clearer detour.

15. **Is the trained model suite stored permanently or retrained on every boot?**  
    *Answer*: It is retrained at startup, which takes a few seconds on 45k rows, so the scores shown always match the code and data. Retraining after that happens only when the user asks for it on the `/retrain` page.

---

## 6. Conclusion
The project ties together search, supervised and unsupervised learning, and model evaluation on real Kaggle delivery data. On a held-out test set the random forest reaches 90.87% accuracy and 0.966 AUC, and linear regression predicts delivery time within about 5 minutes on average. Every chart and score in the web app is computed live from the loaded models.
