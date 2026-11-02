"""Feature engineering для PD-модели."""

import pandas as pd

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

AGE_BINS = [18, 25, 35, 45, 55, 65, 100]
AGE_LABELS = ["18-25", "26-35", "36-45", "46-55", "56-65", "66+"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["age_group"] = pd.cut(df["AGE"], bins=AGE_BINS, labels=AGE_LABELS)

    df["avg_bill_amt"] = df[BILL_COLS].mean(axis=1)
    df["avg_pay_amt"] = df[PAY_AMT_COLS].mean(axis=1)
    df["max_delay"] = df[PAY_COLS].max(axis=1)
    df["months_delayed"] = (df[PAY_COLS] > 0).sum(axis=1)

    # доля платежа от суммы счета, с защитой от деления на 0
    bill = df["avg_bill_amt"].to_numpy(dtype=float)
    pay = df["avg_pay_amt"].to_numpy(dtype=float)
    df["pay_to_bill_ratio"] = pd.Series(
        [p / b if b != 0 else 0.0 for p, b in zip(pay, bill)], index=df.index
    )

    return df


def main():
    import sys

    in_path, out_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(in_path)
    df = add_features(df)
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
