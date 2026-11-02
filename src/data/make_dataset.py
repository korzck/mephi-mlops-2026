"""Загрузка и первичная очистка датасета Default of Credit Card Clients."""

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TARGET = "default payment next month"


def download_raw(dest: Path = RAW_DIR / "UCI_Credit_Card.csv") -> Path:
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DATA_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        xls_name = [n for n in zf.namelist() if n.endswith(".xls")][0]
        with zf.open(xls_name) as f:
            df = pd.read_excel(f, header=1)

    df.to_csv(dest, index=False)
    return dest


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    # в датасете встречаются значения EDUCATION/MARRIAGE вне описанных категорий
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    return df


def split_and_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET]
    )
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    return train_df, test_df


def main():
    raw_path = Path(sys.argv[1]) if len(sys.argv) > 1 else download_raw()
    df = pd.read_csv(raw_path)
    df = clean(df)
    split_and_save(df)
    print(f"processed rows: {len(df)}")


if __name__ == "__main__":
    main()
