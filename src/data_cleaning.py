# src/data_cleaning.py
"""
Reusable preprocessing utilities for tabular ML projects.

Use:
    from src.data_cleaning import infer_feature_types, make_preprocessor, build_preprocessor
"""

from typing import List, Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[List[str], List[str]]:
    """
    Infer which columns are numeric vs categorical, excluding the target.

    Returns:
        num_cols: list of numeric feature names
        cat_cols: list of categorical feature names
    """
    X = df.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    return num_cols, cat_cols


def make_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
      - imputes missing numeric values with median and scales them
      - imputes missing categorical values with the most frequent value and one-hot encodes them
    """
    numeric_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )

    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    """
    Convenience wrapper: infer types and return a ready-to-fit preprocessor.
    """
    num_cols, cat_cols = infer_feature_types(df, target_col)
    return make_preprocessor(num_cols, cat_cols)
