"""Валидация данных с помощью Great Expectations."""

import great_expectations as gx
import pandas as pd

TARGET = "default payment next month"


def build_suite() -> "gx.ExpectationSuite":
    suite = gx.ExpectationSuite(name="credit_data_suite")

    required_columns = ["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE", TARGET]
    for col in required_columns:
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column=col))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="AGE", min_value=18, max_value=100)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(column="LIMIT_BAL", min_value=0)
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column="SEX", value_set=[1, 2])
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1])
    )

    return suite


def validate(df: pd.DataFrame):
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("pandas")
    asset = data_source.add_dataframe_asset(name="credit_data")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = build_suite()
    results = batch.validate(suite)
    return results


def main():
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.csv"
    df = pd.read_csv(path)
    results = validate(df)

    if not results.success:
        print(results)
        raise SystemExit("data validation failed")

    print(f"validation ok, {len(df)} rows checked")


if __name__ == "__main__":
    main()
