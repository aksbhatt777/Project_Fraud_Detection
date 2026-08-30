"""
Data Persistence component.

Mirrors notebook cells 129-139:
    1. Save df_fe as-is (still has some NaN, e.g. card_amt_std for
       single-transaction cards) -> used by tree models (LightGBM/XGBoost
       handle NaN natively) => "fraud_data_clean_forML.parquet"
    2. Fill the remaining NaN (card_amt_std -> 0), since neural nets can't
       handle NaN => "fraud_data_clean.parquet" / ".csv"
"""

import os
import sys

import pandas as pd

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


class DataPersistence:
    def __init__(self, data_dir: str, for_ml_filename: str, no_nan_parquet_filename: str, no_nan_csv_filename: str):
        self.data_dir = data_dir
        self.for_ml_filename = for_ml_filename
        self.no_nan_parquet_filename = no_nan_parquet_filename
        self.no_nan_csv_filename = no_nan_csv_filename

    def save(self, df_fe: pd.DataFrame) -> dict:
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            # 1. Save the ML-ready version (NaN allowed) - notebook cell 129
            ml_path = os.path.join(self.data_dir, self.for_ml_filename)
            df_fe.to_parquet(ml_path, index=False)
            logger.info(f"Saved ML-ready data (NaN allowed): {df_fe.shape} -> {ml_path}")

            # 2. Fill remaining NaN for the NN-ready version - notebook cells 133-139
            nan_cols = df_fe.columns[df_fe.isnull().any()].tolist()
            logger.info(f"Columns with NaN before NN-ready fill: {nan_cols}")

            df_clean = df_fe.copy()
            if "card_amt_std" in df_clean.columns:
                df_clean["card_amt_std"] = df_clean["card_amt_std"].fillna(0)

            nn_parquet_path = os.path.join(self.data_dir, self.no_nan_parquet_filename)
            df_clean.to_parquet(nn_parquet_path, index=False)
            logger.info(f"Saved NN-ready data (parquet, NaN-free): {df_clean.shape} -> {nn_parquet_path}")

            nn_csv_path = os.path.join(self.data_dir, self.no_nan_csv_filename)
            df_clean.to_csv(nn_csv_path, index=False)
            logger.info(f"Saved NN-ready data (csv, NaN-free): {df_clean.shape} -> {nn_csv_path}")

            return {
                "for_ml_path": ml_path,
                "no_nan_parquet_path": nn_parquet_path,
                "no_nan_csv_path": nn_csv_path,
            }
        except Exception as e:
            raise FraudDetectionError(e, sys)
