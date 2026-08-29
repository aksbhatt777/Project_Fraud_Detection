"""
Model Evaluation component.

Mirrors notebook cells 154-163:
    - precision_recall_curve -> F1-optimal threshold
    - accuracy / precision / recall / f1 / confusion matrix / classification report
    - metrics at a scan of thresholds
    - final predictions saved at a chosen operating threshold
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelEvaluationConfig:
    artifacts_dir: str
    chosen_threshold: float = 0.40
    threshold_scan: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def find_f1_optimal_threshold(self, y_test: pd.Series, final_pred: np.ndarray) -> Dict[str, Any]:
        """notebook cells 155-156."""
        precisions, recalls, thresholds = precision_recall_curve(y_test, final_pred)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        # last precision/recall pair has no corresponding threshold, so drop it
        f1_scores = np.nan_to_num(f1_scores[:-1])
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        logger.info(f"Best threshold for F1: {best_threshold:.4f} (F1={best_f1:.4f})")
        return {"threshold": best_threshold, "f1": best_f1, "precisions": precisions, "recalls": recalls, "thresholds": thresholds}

    def evaluate_at_threshold(self, y_test: pd.Series, final_pred: np.ndarray, threshold: float) -> Dict[str, Any]:
        """notebook cells 157-159."""
        y_pred = (final_pred >= threshold).astype(int)

        metrics = {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, final_pred),
        }

        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        logger.info(
            f"@threshold={threshold:.2f}  "
            f"Acc={metrics['accuracy']:.4f}  Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}"
        )
        logger.info(
            f"Confusion Matrix -> TN={cm[0, 0]} FP={cm[0, 1]} FN={cm[1, 0]} TP={cm[1, 1]}"
        )
        logger.info(f"\n{report}")

        return {"metrics": metrics, "confusion_matrix": cm, "classification_report": report, "y_pred": y_pred}

    def scan_thresholds(self, y_test: pd.Series, final_pred: np.ndarray) -> pd.DataFrame:
        """notebook cell 161 - metrics at a range of thresholds."""
        rows = []
        for threshold in self.config.threshold_scan:
            y_pred = (final_pred >= threshold).astype(int)
            rows.append(
                {
                    "threshold": threshold,
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                }
            )
        scan_df = pd.DataFrame(rows)
        logger.info(f"Threshold scan:\n{scan_df.to_string(index=False)}")
        return scan_df

    def save_final_predictions(
        self, y_test: pd.Series, final_pred: np.ndarray, test_transaction_ids: pd.Series
    ) -> str:
        """notebook cell 163 - predictions at the chosen operating threshold."""
        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)
            y_pred_final = (final_pred >= self.config.chosen_threshold).astype(int)

            final_submission = pd.DataFrame(
                {
                    "TransactionID": test_transaction_ids.values,
                    "fraud_probability": final_pred,
                    "predicted_fraud": y_pred_final,
                }
            )
            path = os.path.join(self.config.artifacts_dir, "final_predictions_with_threshold.csv")
            final_submission.to_csv(path, index=False)

            logger.info(
                f"Saved predictions with threshold {self.config.chosen_threshold} -> {path} "
                f"(flagged {y_pred_final.sum()} / {len(y_pred_final)} transactions)"
            )
            return path
        except Exception as e:
            raise FraudDetectionError(e, sys)
