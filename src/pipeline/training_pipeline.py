"""
Training Pipeline.

Orchestrates every component in the same order as the source notebook:

    1. Data Ingestion         (load + merge identity/transaction)
    2. Data Cleaning          (missing value handling)
    3. Feature Engineering    (pass 1 -> baseline features)
    4. Feature Engineering 2.0(pass 2 -> card age, device brand, email groups)
    5. Data Persistence       (save ML-ready + NN-ready artifacts)
    6. Data Split             (time-based, on the ML-ready data)
    7. Model Training         (LightGBM + XGBoost)
    8. [optional] NN Training (on the NN-ready, NaN-free data)
    9. Ensemble               (weighted average of LightGBM + XGBoost)
   10. Model Evaluation       (threshold tuning, final metrics, final CSV)
"""

import sys

from src.components.data_cleaning import DataCleaning, DataCleaningConfig
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_persistence import DataPersistence
from src.components.data_splitter import DataSplitter, DataSplitterConfig
from src.components.ensemble import Ensemble, EnsembleConfig
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig
from src.components.model_evaluation import ModelEvaluation, ModelEvaluationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exceptions import FraudDetectionError
from src.logger import get_logger
from src.utils import read_yaml

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.cfg = read_yaml(config_path)
        self.artifacts_dir = self.cfg["data"]["artifacts_dir"]

    def run(self) -> dict:
        try:
            logger.info("########## TRAINING PIPELINE STARTED ##########")

            # 1. Ingestion --------------------------------------------------
            ingestion = DataIngestion(
                DataIngestionConfig(
                    identity_path=self.cfg["data"]["identity_path"],
                    transaction_path=self.cfg["data"]["transaction_path"],
                )
            )
            df = ingestion.load_and_merge()

            # 2. Cleaning -----------------------------------------------------
            cleaning = DataCleaning(
                DataCleaningConfig(
                    special_cols=self.cfg["cleaning"]["special_cols"],
                    drop_missing_pct_threshold=self.cfg["cleaning"]["drop_missing_pct_threshold"],
                    flag_missing_pct_low=self.cfg["cleaning"]["flag_missing_pct_low"],
                    flag_missing_pct_high=self.cfg["cleaning"]["flag_missing_pct_high"],
                    numeric_fill_value=self.cfg["cleaning"]["numeric_fill_value"],
                    categorical_fill_value=self.cfg["cleaning"]["categorical_fill_value"],
                    v_col_fill_value=self.cfg["cleaning"]["v_col_fill_value"],
                )
            )
            df = cleaning.clean(df)

            # 3 & 4. Feature engineering (pass 1 + pass 2 / 2.0) -------------
            fe_cfg = FeatureEngineeringConfig(
                v_groups=self.cfg["feature_engineering"]["v_groups"],
                high_cardinality_freq_cols=self.cfg["feature_engineering"]["high_cardinality_freq_cols"],
                common_email_domains=self.cfg["feature_engineering"]["common_email_domains"],
                suspicious_email_domains=self.cfg["feature_engineering"]["suspicious_email_domains"],
            )
            feature_engineering = FeatureEngineering(fe_cfg)
            df = feature_engineering.run_first_pass(df)
            df_fe = feature_engineering.run_second_pass(df)

            # 5. Persistence --------------------------------------------------
            persistence = DataPersistence(
                artifacts_dir=self.artifacts_dir,
                for_ml_filename=self.cfg["data"]["clean_for_ml_parquet"],
                no_nan_parquet_filename=self.cfg["data"]["clean_no_nan_parquet"],
                no_nan_csv_filename=self.cfg["data"]["clean_no_nan_csv"],
            )
            saved_paths = persistence.save(df_fe)

            # 6. Split (ML-ready / NaN-allowed data, as train_ML.py does) ----
            splitter = DataSplitter(
                DataSplitterConfig(
                    train_ratio=self.cfg["split"]["train_ratio"],
                    sort_col=self.cfg["split"]["sort_col"],
                    target_col=self.cfg["split"]["target_col"],
                    drop_cols_from_features=self.cfg["split"]["drop_cols_from_features"],
                )
            )
            X_train, y_train, X_test, y_test, train_df, test_df = splitter.split(df_fe)

            # 7. Model training (LightGBM + XGBoost) --------------------------
            trainer = ModelTrainer(
                ModelTrainerConfig(
                    artifacts_dir=self.artifacts_dir,
                    random_state=self.cfg["model_trainer"]["random_state"],
                    lightgbm_params=self.cfg["model_trainer"]["lightgbm"],
                    xgboost_params=self.cfg["model_trainer"]["xgboost"],
                )
            )
            train_results = trainer.train_all(
                X_train, y_train, X_test, y_test, test_df["TransactionID"]
            )

            # 8. [optional] NN training ---------------------------------------
            nn_results = None
            if self.cfg["nn_trainer"]["enabled"]:
                logger.info("nn_trainer.enabled=true -> training neural network")
                from src.components.nn_trainer import NNTrainer, NNTrainerConfig

                # NN uses the NaN-free dataset, split the same time-based way
                import pandas as pd

                df_nn = pd.read_parquet(saved_paths["no_nan_parquet_path"])
                nn_split = DataSplitter(
                    DataSplitterConfig(
                        train_ratio=self.cfg["split"]["train_ratio"],
                        sort_col=self.cfg["split"]["sort_col"],
                        target_col=self.cfg["split"]["target_col"],
                        drop_cols_from_features=self.cfg["split"]["drop_cols_from_features"],
                    )
                )
                nn_X_train, nn_y_train, nn_X_test, nn_y_test, _, nn_test_df = nn_split.split(df_nn)

                nn_cfg = self.cfg["nn_trainer"]
                nn_trainer = NNTrainer(
                    NNTrainerConfig(
                        artifacts_dir=self.artifacts_dir,
                        val_split_ratio=nn_cfg["val_split_ratio"],
                        class_weight_cap=nn_cfg["class_weight_cap"],
                        clip_range=nn_cfg["clip_range"],
                        dense_units=nn_cfg["architecture"]["dense_units"],
                        dropout=nn_cfg["architecture"]["dropout"],
                        l2_reg=nn_cfg["architecture"]["l2_reg"],
                        learning_rate=nn_cfg["optimizer"]["learning_rate"],
                        clipnorm=nn_cfg["optimizer"]["clipnorm"],
                        epochs=nn_cfg["training"]["epochs"],
                        batch_size=nn_cfg["training"]["batch_size"],
                        early_stopping_patience=nn_cfg["training"]["early_stopping_patience"],
                        monitor=nn_cfg["training"]["monitor"],
                    )
                )
                nn_results = nn_trainer.train(
                    nn_X_train, nn_y_train, nn_X_test, nn_y_test, nn_test_df["TransactionID"]
                )

            # 9. Ensemble (LightGBM + XGBoost) ---------------------------------
            ensemble = Ensemble(
                EnsembleConfig(
                    artifacts_dir=self.artifacts_dir,
                    lgb_weight=self.cfg["ensemble"]["lgb_weight"],
                    xgb_weight=self.cfg["ensemble"]["xgb_weight"],
                    weighted_search_start=self.cfg["ensemble"]["weighted_search_range"][0],
                    weighted_search_stop=self.cfg["ensemble"]["weighted_search_range"][1],
                    weighted_search_step=self.cfg["ensemble"]["weighted_search_range"][2],
                )
            )
            lgb_pred = train_results["predictions"]["LightGBM"]
            xgb_pred = train_results["predictions"]["XGBoost"]
            ensemble.run_experiments(lgb_pred, xgb_pred, y_test)
            ensemble_result = ensemble.apply_final_ensemble(
                lgb_pred, xgb_pred, y_test, test_df["TransactionID"]
            )

            # 10. Evaluation ----------------------------------------------------
            evaluator = ModelEvaluation(
                ModelEvaluationConfig(
                    artifacts_dir=self.artifacts_dir,
                    chosen_threshold=self.cfg["evaluation"]["chosen_threshold"],
                    threshold_scan=self.cfg["evaluation"]["threshold_scan"],
                )
            )
            final_pred = ensemble_result["final_pred"]
            f1_opt = evaluator.find_f1_optimal_threshold(y_test, final_pred)
            eval_at_chosen = evaluator.evaluate_at_threshold(
                y_test, final_pred, self.cfg["evaluation"]["chosen_threshold"]
            )
            evaluator.scan_thresholds(y_test, final_pred)
            evaluator.save_final_predictions(y_test, final_pred, test_df["TransactionID"])

            logger.info("########## TRAINING PIPELINE COMPLETE ##########")

            return {
                "train_results": train_results,
                "nn_results": nn_results,
                "ensemble_result": ensemble_result,
                "f1_optimal_threshold": f1_opt,
                "evaluation_at_chosen_threshold": eval_at_chosen,
                "saved_data_paths": saved_paths,
            }
        except Exception as e:
            raise FraudDetectionError(e, sys)
