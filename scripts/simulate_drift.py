"""Имитация потока новых данных и расчет PSI относительно train выборки."""

import numpy as np
import pandas as pd
import requests

from src.data.make_dataset import TARGET
from src.features.build_features import add_features

API_URL = "http://127.0.0.1:8000/predict"


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, breakpoints)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def get_predictions(df: pd.DataFrame) -> np.ndarray:
    probas = []
    for _, row in df.iterrows():
        payload = row.drop(labels=[TARGET], errors="ignore").to_dict()
        resp = requests.post(API_URL, json=payload, timeout=10)
        resp.raise_for_status()
        probas.append(resp.json()["default_probability"])
    return np.array(probas)


def main(sample_size: int = 200):
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv").sample(sample_size, random_state=42)

    train_probas = get_predictions(add_features(train_df).sample(sample_size, random_state=42))
    new_probas = get_predictions(add_features(test_df))

    score = psi(train_probas, new_probas)
    print(f"PSI: {score:.4f}")

    if score < 0.1:
        print("дрифта не обнаружено")
    elif score < 0.2:
        print("умеренный дрифт")
    else:
        print("сильный дрифт, стоит проверить модель")


if __name__ == "__main__":
    main()
