"""
Ensemble component.

Mirrors notebook cells 143-153. The notebook experiments with several
ensembling strategies over LightGBM + XGBoost (+ Random Forest) predictions,
then settles on a weighted average of LightGBM (0.75) + XGBoost (0.25) as
the final ensemble. The Random Forest branch is dropped here since this
project doesn't train Random Forest (see model_trainer.py docstring).
"""

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    artifacts_dir: str
    lgb_weight: float = 0.75
    xgb_weight: float = 0.25
    weighted_search_start: float = 0.5
    weighted_search_stop: float = 0.85
    weighted_search_step: float = 0.05


class Ensemble:
    def __init__(self, config: EnsembleConfig):
        self.config = config

    def run_experiments(self, lgb_pred: np.ndarray, xgb_pred: np.ndarray, y_test: pd.Series) -> Dict[str, Any]:
        """notebook cell 143: simple average / weighted grid / rank average."""
        try:
            logger.info("=" * 60)
            logger.info("ENSEMBLE EXPERIMENTS")
            logger.info("=" * 60)

            # 1. Simple average
            ensemble_avg = (lgb_pred + xgb_pred) / 2
            auc_avg = roc_auc_score(y_test, ensemble_avg)
            logger.info(f"1. Simple Average (50/50): {auc_avg:.4f}")

            # 2. Weighted average grid search
            best_auc, best_weight = 0.0, 0.0
            for w in np.arange(
                self.config.weighted_search_start,
                self.config.weighted_search_stop,
                self.config.weighted_search_step,
            ):
                ensemble_w = w * lgb_pred + (1 - w) * xgb_pred
                auc_w = roc_auc_score(y_test, ensemble_w)
                logger.info(f"   LGB weight {w:.2f}, XGB weight {1 - w:.2f}: AUC = {auc_w:.4f}")
                if auc_w > best_auc:
                    best_auc, best_weight = auc_w, w
            logger.info(f"   Best: LGB {best_weight:.2f} + XGB {1 - best_weight:.2f} = {best_auc:.4f}")

            # 3. Rank average
            lgb_rank = rankdata(lgb_pred) / len(lgb_pred)
            xgb_rank = rankdata(xgb_pred) / len(xgb_pred)
            ensemble_rank = (lgb_rank + xgb_rank) / 2
            auc_rank = roc_auc_score(y_test, ensemble_rank)
            logger.info(f"3. Rank Average: {auc_rank:.4f}")

            logger.info("=" * 60)
            logger.info("ENSEMBLE SUMMARY")
            logger.info(f"Best Ensemble found: {max(auc_avg, best_auc, auc_rank):.4f}")
            logger.info("=" * 60)

            return {
                "simple_average_auc": auc_avg,
                "best_weighted_auc": best_auc,
                "best_weight": best_weight,
                "rank_average_auc": auc_rank,
            }
        except Exception as e:
            raise FraudDetectionError(e, sys)

    def apply_final_ensemble(
        self,
        lgb_pred: np.ndarray,
        xgb_pred: np.ndarray,
        y_test: pd.Series,
        test_transaction_ids: pd.Series,
    ) -> Dict[str, Any]:
        """
        notebook cells 144-152: apply the chosen weighted ensemble
        (0.75 LightGBM + 0.25 XGBoost by default, configurable), evaluate,
        and persist predictions + ensemble metadata.
        """
        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)

            final_pred = self.config.lgb_weight * lgb_pred + self.config.xgb_weight * xgb_pred
            auc = roc_auc_score(y_test, final_pred)
            logger.info(f"Final Ensemble AUC: {auc:.4f}")

            final_predictions = pd.DataFrame(
                {"TransactionID": test_transaction_ids.values, "isFraud": final_pred}
            )
            preds_path = os.path.join(self.config.artifacts_dir, "final_predictions.csv")
            final_predictions.to_csv(preds_path, index=False)

            import joblib

            ensemble_info = {
                "LightGBM_weight": self.config.lgb_weight,
                "XGBoost_weight": self.config.xgb_weight,
                "Test_AUC": auc,
            }
            info_path = os.path.join(self.config.artifacts_dir, "ensemble_info.pkl")
            joblib.dump(ensemble_info, info_path)

            logger.info(f"Saved final predictions -> {preds_path}, ensemble info -> {info_path}")

            return {"final_pred": final_pred, "auc": auc, "ensemble_info": ensemble_info}
        except Exception as e:
            raise FraudDetectionError(e, sys)
