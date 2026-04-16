import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the dataset into training and test sets in a reusable way."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify_option = y if stratify and len(np.unique(y)) > 1 else None
    logger.info(
        "Splitting data: test_size=%s, random_state=%s, stratify=%s",
        test_size,
        random_state,
        stratify_option is not None,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_option,
    )

    logger.info(
        "Train split: %d rows, Test split: %d rows",
        X_train.shape[0],
        X_test.shape[0],
    )
    return X_train, X_test, y_train, y_test


def get_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return explicit numeric and categorical feature lists when available."""
    numeric_columns = ["age", "bmi", "children", "charges"]
    categorical_columns = ["sex", "smoker", "region"]

    if set(numeric_columns + categorical_columns).issubset(df.columns):
        logger.info(
            "Using explicit feature groups: %s numeric, %s categorical",
            numeric_columns,
            categorical_columns,
        )
        return numeric_columns, categorical_columns

    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    logger.info(
        "Detected %d numeric columns and %d categorical columns",
        len(numeric_columns),
        len(categorical_columns),
    )
    return numeric_columns, categorical_columns


def build_preprocessing_pipeline(
    numeric_columns: List[str], categorical_columns: List[str]
) -> ColumnTransformer:
    """Build a reusable sklearn pipeline for imputation, encoding, and scaling."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    logger.info(
        "Built preprocessing pipeline for %d numeric and %d categorical columns",
        len(numeric_columns),
        len(categorical_columns),
    )
    return column_transformer


def fit_preprocessing_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray]:
    """Fit the preprocessing pipeline on training data and transform both train/test splits."""
    numeric_columns, categorical_columns = get_feature_groups(X_train)
    pipeline = build_preprocessing_pipeline(numeric_columns, categorical_columns)

    logger.info("Fitting preprocessing pipeline on training data")
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    return pipeline, X_train_transformed, X_test_transformed
