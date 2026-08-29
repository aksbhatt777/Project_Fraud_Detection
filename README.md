# Fraud Detection — Modular Training Pipeline

A modular, production-style conversion of the IEEE-CIS Fraud Detection
notebook (`FraudDetection.ipynb`) + `train_ML.py` + `train_NN.py` into a
config-driven, logged, exception-safe Python project.

## Project structure

```
fraud_detection/
├── config/
│   └── config.yaml              # every path, threshold, and hyperparameter
├── data/raw/                    # put train_identity.csv, train_transaction.csv here
├── src/
│   ├── logger.py                # logging setup (stdout + logs/*.log)
│   ├── exceptions.py            # FraudDetectionError wrapper
│   ├── utils.py                 # read_yaml, save/load_object, reduce_memory_auto
│   ├── components/
│   │   ├── data_ingestion.py        # load + merge identity/transaction
│   │   ├── data_cleaning.py         # missing value handling
│   │   ├── feature_engineering.py   # V-col aggregates, datetime, card/amount, freq enc, FE 2.0
│   │   ├── data_persistence.py      # save ML-ready (NaN ok) + NN-ready (no NaN) datasets
│   │   ├── data_splitter.py         # time-based train/test split
│   │   ├── model_trainer.py         # LightGBM + XGBoost (train_ML.py)
│   │   ├── nn_trainer.py            # optional Keras NN (train_NN.py)
│   │   ├── ensemble.py              # weighted/rank ensemble experiments + final ensemble
│   │   └── model_evaluation.py      # threshold tuning, metrics, confusion matrix
│   └── pipeline/
│       └── training_pipeline.py     # orchestrates all of the above, in notebook order
├── artifacts/                   # models, predictions, comparison CSVs land here
├── logs/                        # timestamped run logs
├── main.py                      # entry point
├── requirements.txt
└── setup.py
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# place the two source CSVs here (paths configurable in config/config.yaml):
#   data/raw/train_identity.csv
#   data/raw/train_transaction.csv
```

## Run

```bash
python main.py
# or with a different config:
python main.py --config config/config.yaml
```

This runs, in order: ingestion → cleaning → feature engineering (pass 1 +
Feature Engineering 2.0) → save ML-ready/NN-ready datasets → time-based
split → train LightGBM + XGBoost → ensemble → threshold tuning & final
evaluation. Everything lands in `artifacts/`, every step is logged to both
stdout and a timestamped file in `logs/`.

## Fidelity notes — read this if comparing against the notebook

This project deliberately mirrors the notebook / scripts almost line-for-line
(same hyperparameters, same imputation values, same feature formulas, same
ensemble weights) rather than "improving" anything. A few explicit decisions
made when going from notebook → modules, per your instructions:

- **Models trained: LightGBM + XGBoost only.** `train_ML.py` also trained
  Random Forest and Logistic Regression, but both are excluded here — the
  notebook's own comparison found them weaker, and the notebook's *final*
  ensemble (cells 144–152) only ever combines LightGBM + XGBoost anyway.
- **Neural network is optional (`nn_trainer.enabled: false` by default).**
  `train_NN.py`'s logic is preserved as-is in `nn_trainer.py`, but its
  predictions were never part of the notebook's final chosen ensemble, and
  it needs TensorFlow which the rest of the pipeline doesn't require. Flip
  `nn_trainer.enabled: true` in `config.yaml` to run it too.
- **One exploratory step intentionally omitted:** the notebook trains a
  throwaway baseline LightGBM (cells 111–116) on *pass-1* features purely
  to eyeball feature importances before Feature Engineering 2.0. It's never
  saved or used downstream — only the model trained on the final
  (post-FE-2.0) feature set feeds the real artifacts — so it isn't
  reproduced as a pipeline step. Let me know if you want it added back in
  as a diagnostic-only stage.
- **Scope = training pipeline only** (per your answer): no inference/serving
  layer yet. That's a clean follow-on once you're ready to extend to (b)/(c).

## Extending later

- **Batch inference**: reuse `data_cleaning.py` + `feature_engineering.py`
  (same transforms) + `model_trainer`'s saved `.pkl` files in a new
  `prediction_pipeline.py`.
- **API**: wrap the same prediction pipeline in FastAPI once you're ready.
