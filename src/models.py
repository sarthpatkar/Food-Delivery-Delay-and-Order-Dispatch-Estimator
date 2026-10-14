from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE


class DeliveryModelSuite:
    def __init__(self):
        self.regression_model = None
        self.tree_gini_model = None
        self.tree_entropy_model = None
        self.rf_model = None
        self.logistic_model = None
        self.feature_names = []

    def train(self, X_train, y_train_reg, y_train_clf, feature_names):
        self.feature_names = feature_names

        self.regression_model = LinearRegression()
        self.regression_model.fit(X_train, y_train_reg)

        self.tree_gini_model = DecisionTreeClassifier(
            criterion="gini", max_depth=5, min_samples_split=20, random_state=RANDOM_STATE
        )
        self.tree_gini_model.fit(X_train, y_train_clf)

        self.tree_entropy_model = DecisionTreeClassifier(
            criterion="entropy", max_depth=5, min_samples_split=20, random_state=RANDOM_STATE
        )
        self.tree_entropy_model.fit(X_train, y_train_clf)

        self.rf_model = RandomForestClassifier(
            n_estimators=40, max_depth=6, random_state=RANDOM_STATE, n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train_clf)

        self.logistic_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        self.logistic_model.fit(X_train, y_train_clf)

    def predict_sample(self, sample_transformed_df):
        predicted_minutes = float(self.regression_model.predict(sample_transformed_df)[0])
        predicted_minutes = max(8.0, round(predicted_minutes, 1))

        rf_prob = float(self.rf_model.predict_proba(sample_transformed_df)[0][1])
        dt_prob = float(self.tree_gini_model.predict_proba(sample_transformed_df)[0][1])
        lr_prob = float(self.logistic_model.predict_proba(sample_transformed_df)[0][1])

        ensemble_prob = round((0.5 * rf_prob + 0.3 * dt_prob + 0.2 * lr_prob) * 100, 1)

        if ensemble_prob >= 55.0:
            risk_tier = "High"
        elif ensemble_prob >= 30.0:
            risk_tier = "Moderate"
        else:
            risk_tier = "Low"

        return {
            "predicted_minutes": predicted_minutes,
            "delay_probability_pct": ensemble_prob,
            "delay_risk_tier": risk_tier,
            "tree_gini_prediction": int(self.tree_gini_model.predict(sample_transformed_df)[0]),
            "tree_entropy_prediction": int(self.tree_entropy_model.predict(sample_transformed_df)[0]),
            "rf_prediction": int(self.rf_model.predict(sample_transformed_df)[0]),
            "logistic_prediction": int(self.logistic_model.predict(sample_transformed_df)[0]),
        }

    def get_decision_tree_rules(self, max_depth=3):
        return export_text(
            self.tree_gini_model,
            feature_names=self.feature_names,
            max_depth=max_depth,
        )

    def get_feature_importances(self):
        importances = self.rf_model.feature_importances_
        sorted_pairs = sorted(
            zip(self.feature_names, importances), key=lambda x: x[1], reverse=True
        )
        return [{"feature": f, "importance": round(float(v), 4)} for f, v in sorted_pairs]
