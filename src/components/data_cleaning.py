"""
Data Cleaning component - Missing value handling.

Mirrors notebook cells 19-54, in order:
    1. Drop columns with >99% missing values.
    2. For columns with 80-99% missing (excluding special/V columns),
       create a `has_<col>` presence flag, then drop the original 80-99% cols.
    3. Impute what's left, by column family:
         - regular numerical columns -> -999
         - categorical columns       -> 'unknown'
         - V columns                 -> 0
    4. Assert no missing values remain.
"""

import sys
from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.exceptions import FraudDetectionError
from src.logger import get_logger
from src.utils import is_categorical_dtype_col

logger = get_logger(__name__)


@dataclass
class DataCleaningConfig:
    special_cols: List[str] = field(default_factory=lambda: ["TransactionID", "isFraud", "TransactionDT"])
    drop_missing_pct_threshold: float = 99
    flag_missing_pct_low: float = 80
    flag_missing_pct_high: float = 99
    numeric_fill_value: float = -999
    categorical_fill_value: str = "unknown"
    v_col_fill_value: float = 0


class DataCleaning:
    def __init__(self, config: DataCleaningConfig):
        self.config = config

    @staticmethod
    def _split_columns(df: pd.DataFrame):
        """
        Same categorical/numerical split logic used throughout the notebook
        (`df[col].dtype == 'object'`), made robust to pandas' newer string
        dtype (pandas >= 2.x with `infer_string`, default in pandas 3.0),
        under which text CSV columns load as `StringDtype` rather than
        `object` and would otherwise be silently misclassified as numeric.
        """
        categorical_col = [col for col in df.columns if is_categorical_dtype_col(df[col])]
        numerical_col = [col for col in df.columns if not is_categorical_dtype_col(df[col])]
        return categorical_col, numerical_col

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cells 25-34: drop columns with >99% missing."""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        cols_drop_99 = missing_pct[missing_pct > self.config.drop_missing_pct_threshold].index.tolist()
        logger.info(f"Dropping {len(cols_drop_99)} columns with >{self.config.drop_missing_pct_threshold}% missing")
        df = df.drop(columns=cols_drop_99)
        return df

    def _flag_and_drop_moderate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cells 38-41: 80-99% missing -> presence flag, then drop."""
        _, numerical_col = self._split_columns(df)
        v_cols = [col for col in numerical_col if col.startswith("V")]
        missing_pct = (df.isnull().sum() / len(df)) * 100

        cols_80_99 = missing_pct[
            (missing_pct > self.config.flag_missing_pct_low)
            & (missing_pct <= self.config.flag_missing_pct_high)
        ].index.tolist()

        n_flags = 0
        for col in cols_80_99:
            if col not in self.config.special_cols and col not in v_cols:
                df[f"has_{col}"] = df[col].notnull().astype(int)
                n_flags += 1
        logger.info(f"Created {n_flags} presence flags for {self.config.flag_missing_pct_low}-{self.config.flag_missing_pct_high}% missing columns")

        df = df.drop(columns=cols_80_99)
        logger.info(f"Dropped {len(cols_80_99)} moderately-missing original columns. Shape: {df.shape}")
        return df

    def _impute_remaining(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cells 42-53: impute by column family."""
        categorical_cols, numerical_cols = self._split_columns(df)
        v_cols = [col for col in numerical_cols if col.startswith("V")]
        flag_cols = [col for col in numerical_cols if col.startswith("has_")]

        regular_numerical = [
            col for col in numerical_cols
            if col not in v_cols and col not in flag_cols and col not in self.config.special_cols
        ]

        logger.info(f"Imputing {len(regular_numerical)} regular numerical columns with {self.config.numeric_fill_value}")
        for col in regular_numerical:
            df[col] = df[col].fillna(self.config.numeric_fill_value)

        logger.info(f"Imputing {len(categorical_cols)} categorical columns with '{self.config.categorical_fill_value}'")
        for col in categorical_cols:
            df[col] = df[col].fillna(self.config.categorical_fill_value)

        logger.info(f"Filling {len(v_cols)} V columns with {self.config.v_col_fill_value}")
        for col in v_cols:
            df[col] = df[col].fillna(self.config.v_col_fill_value)

        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info(f"Starting cleaning. Input shape: {df.shape}")

            df = self._drop_high_missing_columns(df)
            df = self._flag_and_drop_moderate_missing(df)
            df = self._impute_remaining(df)

            remaining_missing = df.isnull().sum().sum()
            logger.info(f"Remaining missing values: {remaining_missing}. Final shape: {df.shape}")
            assert remaining_missing == 0, "There are still missing values!"
            logger.info("All missing values have been handled.")

            return df
        except Exception as e:
            raise FraudDetectionError(e, sys)
