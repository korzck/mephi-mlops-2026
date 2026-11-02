import pandas as pd

from src.data.validation import validate

GOOD_ROW = {
    "LIMIT_BAL": 20000,
    "AGE": 30,
    "SEX": 1,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "default payment next month": 0,
}


def test_validate_passes_on_clean_data():
    df = pd.DataFrame([GOOD_ROW, {**GOOD_ROW, "AGE": 45}])
    results = validate(df)
    assert results.success


def test_validate_fails_on_anomalous_age():
    bad_row = {**GOOD_ROW, "AGE": 150}
    df = pd.DataFrame([GOOD_ROW, bad_row])
    results = validate(df)
    assert not results.success


def test_validate_fails_on_invalid_target():
    bad_row = {**GOOD_ROW, "default payment next month": 2}
    df = pd.DataFrame([GOOD_ROW, bad_row])
    results = validate(df)
    assert not results.success
