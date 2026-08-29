"""
Data Splitter component.

Mirrors notebook cells 99-104 (and reused identically by train_ML.py /
train_NN.py): a chronological split, NOT a random one, because transactions
are ordered by time and shuffling would leak the future into training.

    df = df.sort_values('TransactionDT')
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]   # older 80%
    test  = df.iloc[split_idx:]   # newer 20%
"""

import sys
from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataSplitterConfig:
    train_ratio: float = 0.8
    sort_col: str = "TransactionDT"
    target_col: str = "isFraud"
    drop_cols_from_features: List[str] = field(
        default_factory=lambda: ["TransactionID", "isFraud", "TransactionDT"]
    )


class DataSplitter:
    def __init__(self, config: DataSplitterConfig):
        self.config = config

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Returns: X_train, y_train, X_test, y_test, train_df, test_df

        `train_df` / `test_df` (the full rows, incl. TransactionID) are
        returned too since downstream steps (predictions CSVs, ensemble)
        need TransactionID alongside the predictions.
        """
        try:
            df = df.sort_values(self.config.sort_col).reset_index(drop=True)
            split_idx = int(len(df) * self.config.train_ratio)

            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

            features = [c for c in df.columns if c not in self.config.drop_cols_from_features]

            X_train = train_df[features]
            y_train = train_df[self.config.target_col]
            X_test = test_df[features]
            y_test = test_df[self.config.target_col]

            logger.info(
                f"Split done. train={X_train.shape}, test={X_test.shape} "
                f"(split_idx={split_idx}, ratio={self.config.train_ratio})"
            )
            logger.info(
                f"Fraud rate - train: {y_train.mean():.4%}, test: {y_test.mean():.4%}"
            )

            return X_train, y_train, X_test, y_test, train_df, test_df
        except Exception as e:
            raise FraudDetectionError(e, sys)
