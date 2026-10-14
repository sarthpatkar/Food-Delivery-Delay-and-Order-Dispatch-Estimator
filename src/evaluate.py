import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.config import RANDOM_STATE


def compute_classification_metrics(model, X_test, y_test, model_name="Model"):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = float(auc(fpr, tpr))

    # Keep about 35 points so the chart payload stays small
    step = max(1, len(fpr) // 35)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), step)
    ]
    if roc_points[-1]["fpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    return {
        "model_name": model_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc_auc, 3),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "roc_curve": roc_points,
    }


def compute_regression_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 3),
    }


def evaluate_suite(model_suite, X_test, y_test_reg, y_test_clf):
    reg_metrics = compute_regression_metrics(
        model_suite.regression_model, X_test, y_test_reg
    )

    models_to_eval = [
        (model_suite.rf_model, "Random forest"),
        (model_suite.tree_gini_model, "Decision tree (Gini)"),
        (model_suite.tree_entropy_model, "Decision tree (entropy)"),
        (model_suite.logistic_model, "Logistic regression"),
    ]

    clf_metrics = []
    for model, name in models_to_eval:
        if model is not None:
            clf_metrics.append(
                compute_classification_metrics(model, X_test, y_test_clf, name)
            )

    return {
        "regression": reg_metrics,
        "classification": clf_metrics,
    }


def run_cross_validation(model, X, y, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return {
        "fold_scores": [round(float(s) * 100, 2) for s in scores],
        "mean_score": round(float(scores.mean()) * 100, 2),
        "std_score": round(float(scores.std()) * 100, 2),
    }
