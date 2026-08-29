"""
Feature Engineering component.

Mirrors notebook cells 56-96 (first pass) and 117-125 (Feature Engineering 2.0),
in the exact order they were executed in the notebook:

  First pass:
    1. Aggregate V-columns into group-level sum/mean/count/max/std, drop originals.
    2. TransactionDT -> hour_of_day, day_of_week, month.
    3. Sort by (card1, TransactionDT); time_since_last_card.
    4. Per-card amount stats (mean/std/min/max) + amount_deviation.
    5. Per-email amount stats (mean/std).
    6. Frequency encoding of high-cardinality columns.
    7. object -> category dtype; numeric memory downcast.

  Feature Engineering 2.0 (applied on a copy, df_fe):
    8. card_txn_count, card_age.
    9. device_brand (from DeviceInfo).
    10. amount_to_mean.
    11. email_domain_grouped (common / suspicious / rare / unknown).
"""

import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

from src.exceptions import FraudDetectionError
from src.logger import get_logger
from src.utils import is_categorical_dtype_col, reduce_memory_auto

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringConfig:
    v_groups: Dict[str, List[int]] = field(default_factory=dict)   # name -> [start, end] inclusive
    high_cardinality_freq_cols: List[str] = field(default_factory=list)
    common_email_domains: List[str] = field(default_factory=list)
    suspicious_email_domains: List[str] = field(default_factory=list)


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    # ---------------------------------------------------------- V columns --
    def _aggregate_v_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cells 60-64."""
        v_groups = {
            name: [f"V{i}" for i in range(bounds[0], bounds[1] + 1)]
            for name, bounds in self.config.v_groups.items()
        }

        for group_name, cols in v_groups.items():
            existing = [c for c in cols if c in df.columns]
            if existing:
                df[f"{group_name}_sum"] = df[existing].sum(axis=1)
                df[f"{group_name}_mean"] = df[existing].mean(axis=1)
                df[f"{group_name}_count"] = (df[existing] > 0).sum(axis=1)
                df[f"{group_name}_max"] = df[existing].max(axis=1)
                df[f"{group_name}_std"] = df[existing].std(axis=1).fillna(0)
                logger.info(f"  {group_name}: {len(existing)} columns -> 5 features")

        # Drop original V columns (matched exactly like the notebook, e.g. V1, V23...)
        v_cols = [col for col in df.columns if re.match(r"^V\d+$", col)]
        df = df.drop(columns=v_cols)
        logger.info(f"Dropped {len(v_cols)} original V columns")
        return df

    # ------------------------------------------------------- datetime feats
    def _add_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cells 67-75."""
        df["hour_of_day"] = (df["TransactionDT"] // 3600) % 24
        df["day_of_week"] = (df["TransactionDT"] // 86400) % 7
        df["month"] = (df["TransactionDT"] // (86400 * 30)) % 12

        df = df.sort_values(["card1", "TransactionDT"])
        df["time_since_last_card"] = df.groupby("card1")["TransactionDT"].diff()
        df["time_since_last_card"] = df["time_since_last_card"].fillna(-1)
        return df

    # --------------------------------------------------------- amount feats
    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cells 78-82."""
        df["card_amt_mean"] = df.groupby("card1")["TransactionAmt"].transform("mean")
        df["card_amt_std"] = df.groupby("card1")["TransactionAmt"].transform("std")
        df["card_amt_min"] = df.groupby("card1")["TransactionAmt"].transform("min")
        df["card_amt_max"] = df.groupby("card1")["TransactionAmt"].transform("max")
        df["amount_deviation"] = df["TransactionAmt"] - df["card_amt_mean"]

        if "P_emaildomain" in df.columns:
            df["email_amt_mean"] = df.groupby("P_emaildomain")["TransactionAmt"].transform("mean")
            df["email_amt_std"] = df.groupby("P_emaildomain")["TransactionAmt"].transform("std")

        return df

    # ------------------------------------------------------ frequency enc.
    def _add_frequency_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """notebook cell 86."""
        for col in self.config.high_cardinality_freq_cols:
            if col in df.columns:
                freq = df[col].value_counts()
                df[f"{col}_freq"] = df[col].map(freq)
        return df

    # ------------------------------------------------------- dtype/memory
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        notebook cells 90-96. Uses `is_categorical_dtype_col` instead of
        `select_dtypes(include=['object'])` for the same pandas-version
        robustness reason documented in `data_cleaning.py`.
        """
        cat_cols = [col for col in df.columns if is_categorical_dtype_col(df[col])]
        for col in cat_cols:
            df[col] = df[col].astype("category")

        df = reduce_memory_auto(df)
        return df

    def run_first_pass(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Feature engineering (pass 1): V-column aggregation")
            df = self._aggregate_v_columns(df)

            logger.info("Feature engineering (pass 1): datetime features")
            df = self._add_datetime_features(df)

            logger.info("Feature engineering (pass 1): amount features")
            df = self._add_amount_features(df)

            logger.info("Feature engineering (pass 1): frequency encoding")
            df = self._add_frequency_encoding(df)

            logger.info("Feature engineering (pass 1): dtype/memory optimization")
            df = self._optimize_dtypes(df)

            return df
        except Exception as e:
            raise FraudDetectionError(e, sys)

    # ===================================================================
    # Feature Engineering 2.0 (notebook cells 117-125)
    # ===================================================================
    def _group_email_domain(self, domain: str) -> str:
        if domain == "unknown":
            return "unknown"
        elif domain in self.config.suspicious_email_domains:
            return "suspicious"
        elif domain in self.config.common_email_domains:
            return domain
        else:
            return "rare_domain"

    def run_second_pass(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature Engineering 2.0 - applied to a copy of the pass-1 dataframe
        (`df_fe = df.copy()` in the notebook), adding card usage/age,
        device brand, amount-to-mean ratio, and grouped email domain.
        """
        try:
            logger.info("Feature engineering (pass 2 / 2.0): card usage & age")
            df_fe = df.copy()
            df_fe = df_fe.sort_values(["card1", "TransactionDT"])
            df_fe["card_txn_count"] = df_fe.groupby("card1").cumcount()

            df_fe["card_first_txn"] = df_fe.groupby("card1")["TransactionDT"].transform("min")
            df_fe["card_age"] = df_fe["TransactionDT"] - df_fe["card_first_txn"]
            df_fe = df_fe.drop(columns=["card_first_txn"])

            logger.info("Feature engineering (pass 2 / 2.0): device brand")
            df_fe["device_brand"] = df_fe["DeviceInfo"].str.split().str[0].fillna("unknown")
            df_fe["device_brand"] = df_fe["device_brand"].astype("category")

            logger.info("Feature engineering (pass 2 / 2.0): amount_to_mean")
            df_fe["amount_to_mean"] = df_fe["TransactionAmt"] / (df_fe["card_amt_mean"] + 1)

            logger.info("Feature engineering (pass 2 / 2.0): grouped email domain")
            df_fe["email_domain_grouped"] = df_fe["P_emaildomain"].apply(self._group_email_domain)
            df_fe["email_domain_grouped"] = df_fe["email_domain_grouped"].astype("category")

            return df_fe
        except Exception as e:
            raise FraudDetectionError(e, sys)
