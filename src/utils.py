"""
Small shared utilities used by multiple components.
"""

import os
import sys
from typing import Any, Dict

import joblib
import pandas as pd
import yaml
from pandas.api.types import is_object_dtype, is_string_dtype

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    try:
        with open(path, "r") as f:
            content = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return content
    except Exception as e:
        raise FraudDetectionError(e, sys)


def create_directories(paths) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_object(obj: Any, file_path: str) -> None:
    """Persist any picklable object (model, dict, etc.) via joblib."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Saved object to {file_path}")
    except Exception as e:
        raise FraudDetectionError(e, sys)


def load_object(file_path: str) -> Any:
    try:
        obj = joblib.load(file_path)
        logger.info(f"Loaded object from {file_path}")
        return obj
    except Exception as e:
        raise FraudDetectionError(e, sys)


def is_categorical_dtype_col(series: pd.Series) -> bool:
    """
    True for text/categorical-like columns. The notebook identifies these
    via `dtype == 'object'`, which is correct on the pandas version the
    notebook was authored on. Newer pandas (>= 2.x with `infer_string`,
    the default from pandas 3.0 onward) instead loads CSV text columns as
    `StringDtype`, which `== 'object'` no longer catches - this helper
    keeps the same intent (treat text columns as categorical) across
    pandas versions.
    """
    return is_object_dtype(series) or is_string_dtype(series)


def reduce_memory_auto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically downcast all numeric columns using pandas built-ins.

    Verbatim logic from notebook cell 94 (`reduce_memory_auto`), just moved
    into the shared utils module since it's a generic helper.
    """
    try:
        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            if df[col].dtype == "int64":
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif df[col].dtype == "float64":
                df[col] = pd.to_numeric(df[col], downcast="float")

        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB")
        return df
    except Exception as e:
        raise FraudDetectionError(e, sys)
