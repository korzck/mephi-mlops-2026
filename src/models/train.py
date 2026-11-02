"""Обучение PD-модели с логированием экспериментов в MLflow."""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.data.make_dataset import TARGET
from src.features.build_features import add_features
from src.models.metrics import compute_metrics, plot_roc_curve
from src.models.pipeline import create_pipeline

MODEL_PATH = Path("models/credit_default_model.pkl")
ROC_PLOT_PATH = Path("models/roc_curve.png")
METRICS_PATH = Path("metrics.json")

EXPERIMENTS = [
    ("logreg_c01", LogisticRegression(max_iter=1000, C=0.1)),
    ("logreg_c1", LogisticRegression(max_iter=1000, C=1.0)),
    ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("gboost_100", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ("gboost_200", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)),
]

GRID_SEARCH_PARAMS = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [2, 3, 4],
}


def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]
    return X_train, X_test, y_train, y_test


def run_experiment(name, classifier, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=name):
        pipeline = create_pipeline(classifier)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_param("model_type", type(classifier).__name__)
        mlflow.log_params(classifier.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        return pipeline, metrics


def run_grid_search(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="gboost_grid_search"):
        base_pipeline = create_pipeline(GradientBoostingClassifier(random_state=42))
        search = GridSearchCV(base_pipeline, GRID_SEARCH_PARAMS, scoring="roc_auc", cv=3, n_jobs=-1)
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        y_proba = best_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_param("model_type", "GradientBoostingClassifier")
        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        roc_path = plot_roc_curve(y_test, y_proba, str(ROC_PLOT_PATH))
        mlflow.log_artifact(roc_path)
        mlflow.sklearn.log_model(best_pipeline, "model", registered_model_name="CreditDefaultModel")

        return best_pipeline, metrics


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("credit_default_prediction")

    X_train, X_test, y_train, y_test = load_data()

    for name, classifier in EXPERIMENTS:
        _, metrics = run_experiment(name, classifier, X_train, X_test, y_train, y_test)
        print(name, metrics)

    best_pipeline, best_metrics = run_grid_search(X_train, X_test, y_train, y_test)
    print("grid search best:", best_metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(best_metrics, f, indent=2)


if __name__ == "__main__":
    main()
