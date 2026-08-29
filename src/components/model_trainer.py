"""
Model Trainer component.

Mirrors train_ML.py, keeping only LightGBM + XGBoost (Random Forest and
Logistic Regression from the original script are intentionally excluded -
the notebook's own model comparison found them weaker, and the notebook's
final ensemble only ever combines LightGBM + XGBoost).
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelTrainerConfig:
    artifacts_dir: str
    random_state: int = 42
    lightgbm_params: Dict[str, Any] = field(default_factory=dict)
    xgboost_params: Dict[str, Any] = field(default_factory=dict)


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    @staticmethod
    def _label_encode_categoricals(
        X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        XGBoost (unlike LightGBM) needs categoricals label-encoded.
        Fit the encoder on train+test combined, exactly as train_ML.py does,
        so unseen test-only categories don't break `.transform`.
        """
        cat_cols = X_train.select_dtypes(include=["category"]).columns
        X_train_num = X_train.copy()
        X_test_num = X_test.copy()

        for col in cat_cols:
            le = LabelEncoder()
            le.fit(pd.concat([X_train[col].astype(str), X_test[col].astype(str)]))
            X_train_num[col] = le.transform(X_train[col].astype(str))
            X_test_num[col] = le.transform(X_test[col].astype(str))

        return X_train_num, X_test_num

    def train_lightgbm(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Any, "pd.Series[float]", float]:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Training LightGBM (scale_pos_weight={scale_pos_weight:.1f})")
        start = time.time()

        model = lgb.LGBMClassifier(
            **self.config.lightgbm_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, categorical_feature="auto")

        preds = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        logger.info(f"LightGBM AUC: {auc:.4f} | Time: {time.time() - start:.1f}s")

        save_path = os.path.join(self.config.artifacts_dir, "best_lightgbm.pkl")
        joblib.dump(model, save_path)
        logger.info(f"Saved LightGBM model to {save_path}")

        return model, preds, auc

    def train_xgboost(
        self, X_train_num: pd.DataFrame, y_train: pd.Series, X_test_num: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Any, "pd.Series[float]", float]:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.1f})")
        start = time.time()

        model = xgb.XGBClassifier(
            **self.config.xgboost_params,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        model.fit(X_train_num, y_train)

        preds = model.predict_proba(X_test_num)[:, 1]
        auc = roc_auc_score(y_test, preds)
        logger.info(f"XGBoost AUC: {auc:.4f} | Time: {time.time() - start:.1f}s")

        save_path = os.path.join(self.config.artifacts_dir, "best_xgboost.pkl")
        joblib.dump(model, save_path)
        logger.info(f"Saved XGBoost model to {save_path}")

        return model, preds, auc

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_transaction_ids: pd.Series,
    ) -> Dict[str, Any]:
        """
        Trains LightGBM + XGBoost, saves models + comparison + predictions
        CSVs, matching train_ML.py's output artifacts:
            - best_lightgbm.pkl / best_xgboost.pkl
            - ml_model_comparison.csv
            - ml_model_test_predictions.csv
        """
        try:
            os.makedirs(self.config.artifacts_dir, exist_ok=True)

            X_train_num, X_test_num = self._label_encode_categoricals(X_train, X_test)

            results: Dict[str, float] = {}
            predictions: Dict[str, Any] = {}
            models: Dict[str, Any] = {}

            lgb_model, lgb_preds, lgb_auc = self.train_lightgbm(X_train, y_train, X_test, y_test)
            results["LightGBM"] = lgb_auc
            predictions["LightGBM"] = lgb_preds
            models["LightGBM"] = lgb_model

            xgb_model, xgb_preds, xgb_auc = self.train_xgboost(X_train_num, y_train, X_test_num, y_test)
            results["XGBoost"] = xgb_auc
            predictions["XGBoost"] = xgb_preds
            models["XGBoost"] = xgb_model

            logger.info("Model comparison (Test AUC):")
            for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {name:<12} AUC: {auc:.4f}")

            results_df = pd.DataFrame(list(results.items()), columns=["Model", "AUC"])
            results_df = results_df.sort_values("AUC", ascending=False)
            results_path = os.path.join(self.config.artifacts_dir, "ml_model_comparison.csv")
            results_df.to_csv(results_path, index=False)

            pred_df = pd.DataFrame({"TransactionID": test_transaction_ids.values})
            for name, pred in predictions.items():
                pred_df[name + "_fraud_prob"] = pred
            preds_path = os.path.join(self.config.artifacts_dir, "ml_model_test_predictions.csv")
            pred_df.to_csv(preds_path, index=False)

            logger.info(f"Saved comparison -> {results_path}, predictions -> {preds_path}")

            return {
                "models": models,
                "results": results,
                "predictions": predictions,
                "predictions_df": pred_df,
                "comparison_df": results_df,
            }
        except Exception as e:
            raise FraudDetectionError(e, sys)
