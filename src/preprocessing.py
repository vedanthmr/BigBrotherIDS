"""Loading and feature preparation for UNSW-NB15 CSV files."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "label"
EXCLUDED_FEATURES = {TARGET_COLUMN, "attack_cat", "id"}


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a labeled CSV and return features and the binary target."""
    frame = pd.read_csv(csv_path)
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(
            f"Expected a '{TARGET_COLUMN}' target column in {csv_path}. "
            "Check that this is a labeled UNSW-NB15 split."
        )

    features = frame.drop(columns=[column for column in EXCLUDED_FEATURES if column in frame])
    target = frame[TARGET_COLUMN]
    if target.isna().any():
        raise ValueError(f"The '{TARGET_COLUMN}' column contains missing values.")
    return features, target


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create imputing and encoding steps based on the feature dtypes."""
    categorical_columns = features.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()

    transformers = []
    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No numeric or categorical feature columns were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
