import pandas as pd
from sklearn.impute import SimpleImputer


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_columns = df.select_dtypes(exclude=["object"]).columns
    categorical_columns = df.select_dtypes(include=["object"]).columns

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    df = pd.get_dummies(df, drop_first=True)

    return df
