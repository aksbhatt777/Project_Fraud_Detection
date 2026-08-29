"""
Neural Network Trainer component (optional).

Mirrors train_NN.py exactly. Kept as a separate, opt-in component
(`nn_trainer.enabled: true` in config.yaml) because:
  - It needs TensorFlow, a much heavier dependency than LightGBM/XGBoost.
  - In the source notebook, the NN's predictions are never part of the
    final chosen ensemble (only LightGBM + XGBoost are), so it isn't on
    the critical path of the "train -> evaluate" pipeline.

TensorFlow is imported lazily inside `train()` so the rest of the project
works even if TensorFlow isn't installed.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.exceptions import FraudDetectionError
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class NNTrainerConfig:
    artifacts_dir: str
    val_split_ratio: float = 0.85
    class_weight_cap: float = 10.0
    clip_range: List[float] = field(default_factory=lambda: [-5, 5])
    dense_units: List[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.2])
    l2_reg: float = 0.001
    learning_rate: float = 0.0001
    clipnorm: float = 1.0
    epochs: int = 100
    batch_size: int = 1024
    early_stopping_patience: int = 10
    monitor: str = "val_auc"


class NNTrainer:
    def __init__(self, config: NNTrainerConfig):
        self.config = config

    @staticmethod
    def _label_encode_categoricals(X_train_full: pd.DataFrame, X_test: pd.DataFrame):
        cat_cols = X_train_full.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            le = LabelEncoder()
            le.fit(pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)]))
            X_train_full[col] = le.transform(X_train_full[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
        return X_train_full, X_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        test_transaction_ids: pd.Series,
    ) -> Dict[str, Any]:
        try:
            from tensorflow import keras  # lazy import - optional dependency

            os.makedirs(self.config.artifacts_dir, exist_ok=True)

            X_train_full = X_train.copy()
            X_test = X_test.copy()

            X_train_full, X_test = self._label_encode_categoricals(X_train_full, X_test)

            X_train_full = X_train_full.astype(np.float32).fillna(0)
            X_test = X_test.astype(np.float32).fillna(0)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_full)
            X_test_scaled = scaler.transform(X_test)

            lo, hi = self.config.clip_range
            X_train_scaled = np.clip(X_train_scaled, lo, hi)
            X_test_scaled = np.clip(X_test_scaled, lo, hi)

            val_split = int(len(X_train_scaled) * self.config.val_split_ratio)
            X_train_nn = X_train_scaled[:val_split]
            y_train_nn = y_train.iloc[:val_split].values
            X_val_nn = X_train_scaled[val_split:]
            y_val_nn = y_train.iloc[val_split:].values

            logger.info(
                f"NN train shape: {X_train_nn.shape}, val shape: {X_val_nn.shape}, "
                f"test shape: {X_test_scaled.shape}"
            )

            scale_pos_weight = (y_train_nn == 0).sum() / (y_train_nn == 1).sum()
            class_weights = {0: 1.0, 1: min(scale_pos_weight, self.config.class_weight_cap)}
            logger.info(f"Class weights: {class_weights}")

            units = self.config.dense_units
            dropout = self.config.dropout
            l2 = keras.regularizers.l2(self.config.l2_reg)

            model = keras.Sequential(
                [
                    keras.layers.Dense(
                        units[0], activation="relu", kernel_regularizer=l2,
                        input_shape=(X_train_scaled.shape[1],),
                    ),
                    keras.layers.Dropout(dropout[0]),
                    keras.layers.Dense(units[1], activation="relu", kernel_regularizer=l2),
                    keras.layers.Dropout(dropout[1]),
                    keras.layers.Dense(units[2], activation="relu"),
                    keras.layers.Dropout(dropout[2]),
                    keras.layers.Dense(1, activation="sigmoid"),
                ]
            )

            optimizer = keras.optimizers.Adam(
                learning_rate=self.config.learning_rate, clipnorm=self.config.clipnorm
            )
            model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(name="auc")],
            )

            early_stopping = keras.callbacks.EarlyStopping(
                monitor=self.config.monitor,
                mode="max",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )

            logger.info("Training Neural Network...")
            start = time.time()
            history = model.fit(
                X_train_nn, y_train_nn,
                validation_data=(X_val_nn, y_val_nn),
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                class_weight=class_weights,
                callbacks=[early_stopping],
                verbose=1,
            )

            logger.info(f"Training stopped at epoch: {len(history.history['loss'])}")
            logger.info(f"Best validation AUC: {max(history.history['val_auc']):.4f}")

            nn_pred = model.predict(X_test_scaled).flatten()
            if np.isnan(nn_pred).sum() > 0:
                raise ValueError("NaN present in NN predictions")

            nn_auc = roc_auc_score(y_test, nn_pred)
            logger.info(f"Neural Network Test AUC: {nn_auc:.4f} | Time: {time.time() - start:.1f}s")

            model_path = os.path.join(self.config.artifacts_dir, "best_neural_network.keras")
            model.save(model_path)

            pred_df = pd.DataFrame(
                {"TransactionID": test_transaction_ids.values, "NN_fraud_prob": nn_pred}
            )
            preds_path = os.path.join(self.config.artifacts_dir, "nn_test_predictions.csv")
            pred_df.to_csv(preds_path, index=False)

            logger.info(f"Saved NN model -> {model_path}, predictions -> {preds_path}")

            return {"model": model, "auc": nn_auc, "predictions": nn_pred, "predictions_df": pred_df}
        except Exception as e:
            raise FraudDetectionError(e, sys)
