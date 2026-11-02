import pandas as pd

from src.features.build_features import add_features


def make_row(**overrides):
    row = {
        "AGE": 30,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 1000,
        "BILL_AMT2": 1000,
        "BILL_AMT3": 1000,
        "BILL_AMT4": 1000,
        "BILL_AMT5": 1000,
        "BILL_AMT6": 1000,
        "PAY_AMT1": 100,
        "PAY_AMT2": 100,
        "PAY_AMT3": 100,
        "PAY_AMT4": 100,
        "PAY_AMT5": 100,
        "PAY_AMT6": 100,
    }
    row.update(overrides)
    return row


def test_age_group_assigned():
    df = pd.DataFrame([make_row(AGE=22), make_row(AGE=50)])
    out = add_features(df)
    assert out["age_group"].tolist() == ["18-25", "46-55"]


def test_avg_columns_computed():
    df = pd.DataFrame([make_row()])
    out = add_features(df)
    assert out["avg_bill_amt"].iloc[0] == 1000
    assert out["avg_pay_amt"].iloc[0] == 100


def test_pay_to_bill_ratio_handles_zero_bill():
    df = pd.DataFrame(
        [
            make_row(
                BILL_AMT1=0,
                BILL_AMT2=0,
                BILL_AMT3=0,
                BILL_AMT4=0,
                BILL_AMT5=0,
                BILL_AMT6=0,
            )
        ]
    )
    out = add_features(df)
    assert out["pay_to_bill_ratio"].iloc[0] == 0


def test_months_delayed_counts_positive_pay_status():
    df = pd.DataFrame([make_row(PAY_0=2, PAY_2=1, PAY_3=-1, PAY_4=0, PAY_5=0, PAY_6=0)])
    out = add_features(df)
    assert out["months_delayed"].iloc[0] == 2
