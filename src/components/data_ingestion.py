"""
Data Ingestion component.

Mirrors notebook cells 4-14:
    df1 = pd.read_csv(train_identity.csv)
    df2 = pd.read_csv(train_transaction.csv)
    df  = df2.merge(df1, on='TransactionID', how='left')
"""

import sys
from dataclasses import dataclass

import pandas as pd

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataIngestionConfig:
    identity_path: str
    transaction_path: str


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def load_and_merge(self) -> pd.DataFrame:
        """Load identity + transaction CSVs and left-merge on TransactionID."""
        try:
            logger.info(f"Reading identity data from {self.config.identity_path}")
            df_identity = pd.read_csv(self.config.identity_path)

            logger.info(f"Reading transaction data from {self.config.transaction_path}")
            df_transaction = pd.read_csv(self.config.transaction_path)

            logger.info(
                f"identity shape={df_identity.shape}, transaction shape={df_transaction.shape}"
            )

            # Same merge as notebook: transaction (left) <- identity (right)
            df = df_transaction.merge(df_identity, on="TransactionID", how="left")
            logger.info(f"Merged dataframe shape: {df.shape}")

            return df
        except Exception as e:
            raise FraudDetectionError(e, sys)
