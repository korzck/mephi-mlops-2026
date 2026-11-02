from src.models.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = [0.1, 0.9, 0.2, 0.8]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_contains_expected_keys():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 0, 1, 0]
    y_proba = [0.2, 0.4, 0.6, 0.3]

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert set(metrics.keys()) == {"roc_auc", "precision", "recall", "f1"}
