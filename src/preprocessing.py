import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. REMOVE DUPLICATE ROWS
# ============================================================

def remove_duplicates(df):
    """
    Remove completely duplicated network-flow records.
    """
    before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    after = len(df)

    print(f"Duplicate rows removed: {before - after}")
    print(f"Rows remaining: {after}")

    return df


# ============================================================
# 2. IDENTIFY NUMERIC COLUMNS
# ============================================================

def get_numeric_columns(df, exclude_columns=None):
    """
    Return numeric columns while excluding target/non-feature columns.
    """

    if exclude_columns is None:
        exclude_columns = []

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_columns = [
        col for col in numeric_columns
        if col not in exclude_columns
    ]

    return numeric_columns


# ============================================================
# 3. IQR OUTLIER HANDLING
# ============================================================

def handle_outliers_iqr(df, numeric_columns, multiplier=1.5):
    """
    Handle extreme numeric values using the IQR method.

    Instead of deleting extreme rows, values outside the
    IQR boundaries are clipped to the lower/upper limits.

    This is safer for IDS because extreme values may represent
    genuine attack traffic.
    """

    df = df.copy()

    for column in numeric_columns:

        if column not in df.columns:
            continue

        # Ignore columns that contain no usable values
        if df[column].dropna().empty:
            continue

        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        df[column] = df[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    print("IQR-based outlier handling completed.")

    return df


# ============================================================
# 4. OPTIONAL Z-SCORE OUTLIER HANDLING
# ============================================================

def handle_outliers_zscore(df, numeric_columns, threshold=3):
    """
    Handle extreme values using Z-score clipping.

    Values with |z| > threshold are clipped to the
    threshold boundary.
    """

    df = df.copy()

    for column in numeric_columns:

        if column not in df.columns:
            continue

        mean = df[column].mean()
        std = df[column].std()

        if pd.isna(std) or std == 0:
            continue

        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        df[column] = df[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

    print("Z-score based outlier handling completed.")

    return df


# ============================================================
# 5. IMPUTATION
# ============================================================

def impute_numeric_features(df, numeric_columns):
    """
    Replace missing numerical values using median imputation.
    """

    df = df.copy()

    imputer = SimpleImputer(strategy="median")

    df[numeric_columns] = imputer.fit_transform(
        df[numeric_columns]
    )

    return df, imputer


# ============================================================
# 6. FEATURE SCALING
# ============================================================

def scale_numeric_features(df, numeric_columns):
    """
    Standardize numerical features.
    """

    df = df.copy()

    scaler = StandardScaler()

    df[numeric_columns] = scaler.fit_transform(
        df[numeric_columns]
    )

    return df, scaler


# ============================================================
# 7. CORRELATION-BASED FEATURE PRUNING
# ============================================================

def correlation_feature_pruning(
    df,
    numeric_columns,
    threshold=0.95
):
    """
    Remove highly correlated redundant features.

    If two features have correlation greater than the threshold,
    one of them is removed.
    """

    correlation_matrix = df[numeric_columns].corr().abs()

    upper_triangle = correlation_matrix.where(
        np.triu(
            np.ones(correlation_matrix.shape),
            k=1
        ).astype(bool)
    )

    columns_to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]

    df = df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    remaining_columns = [
        column
        for column in numeric_columns
        if column not in columns_to_drop
    ]

    print(
        f"Highly correlated features removed: "
        f"{len(columns_to_drop)}"
    )

    print("Removed features:", columns_to_drop)

    return df, remaining_columns


# ============================================================
# 8. MODEL-BASED FEATURE SELECTION
# ============================================================

def select_features_model_based(
    X,
    y,
    threshold="median"
):
    """
    Select important features using Random Forest
    and SelectFromModel.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X, y)

    selector = SelectFromModel(
        model,
        threshold=threshold,
        prefit=True
    )

    X_selected = selector.transform(X)

    selected_features = X.columns[
        selector.get_support()
    ].tolist()

    print(
        f"Features before model selection: {X.shape[1]}"
    )

    print(
        f"Features after model selection: {X_selected.shape[1]}"
    )

    print("Selected features:")
    print(selected_features)

    return (
        pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index
        ),
        selector,
        selected_features
    )