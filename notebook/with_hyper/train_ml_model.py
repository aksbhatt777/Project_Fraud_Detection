# train_ml_model.py
import pandas as pd
import numpy as np
import time
import warnings
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    df = pd.read_parquet('/home/ubuntu/ann/notebook/fraud_data_clean_forML.parquet')
    print(f"Data shape: {df.shape}")

    # Sort by time
    df = df.sort_values('TransactionDT').reset_index(drop=True)

    # Time-based split
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    target = 'isFraud'
    features = [col for col in df.columns if col not in ['TransactionID', 'isFraud', 'TransactionDT']]

    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Fraud rate: {y_train.mean():.4%}")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"scale_pos_weight: {scale_pos_weight:.1f}")

    # Prepare numeric version for sklearn models
    cat_cols = X_train.select_dtypes(include=['category']).columns
    print(f"\nCategorical columns: {len(cat_cols)}")

    X_train_num = X_train.copy()
    X_test_num = X_test.copy()
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train[col].astype(str), X_test[col].astype(str)]))
        X_train_num[col] = le.transform(X_train[col].astype(str))
        X_test_num[col] = le.transform(X_test[col].astype(str))

    results = {}
    predictions = {}

    # ============================================
    # 1. LIGHTGBM
    # ============================================
    print("\n" + "="*60)
    print("TUNING LIGHTGBM")
    print("="*60)
    start = time.time()

    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    )

    param_grid_lgb = {
        'num_leaves': [127, 255],
        'min_child_samples': [20, 50]
    }

    grid_lgb = GridSearchCV(
        lgb_model,
        param_grid=param_grid_lgb,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1  # Show progress
    )
    grid_lgb.fit(X_train, y_train, categorical_feature='auto')

    best_lgb = grid_lgb.best_estimator_
    lgb_pred = best_lgb.predict_proba(X_test)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    results['LightGBM'] = lgb_auc
    predictions['LightGBM'] = lgb_pred
    print(f"Best params: {grid_lgb.best_params_}")
    print(f"AUC: {lgb_auc:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(best_lgb, 'best_lightgbm.pkl')

    # ============================================
    # 2. XGBOOST
    # ============================================
    print("\n" + "="*60)
    print("TUNING XGBOOST")
    print("="*60)
    start = time.time()

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )

    param_grid_xgb = {
        'max_depth': [6, 8],
        'min_child_weight': [1, 5]
    }

    grid_xgb = GridSearchCV(
        xgb_model,
        param_grid=param_grid_xgb,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1  # Show progress
    )
    grid_xgb.fit(X_train_num, y_train)

    best_xgb = grid_xgb.best_estimator_
    xgb_pred = best_xgb.predict_proba(X_test_num)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    results['XGBoost'] = xgb_auc
    predictions['XGBoost'] = xgb_pred
    print(f"Best params: {grid_xgb.best_params_}")
    print(f"AUC: {xgb_auc:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(best_xgb, 'best_xgboost.pkl')

    # ============================================
    # 3. RANDOM FOREST
    # ============================================
    print("\n" + "="*60)
    print("TUNING RANDOM FOREST")
    print("="*60)
    start = time.time()

    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    param_grid_rf = {
        'max_depth': [10, 15],
        'min_samples_split': [10, 20]
    }

    grid_rf = GridSearchCV(
        rf_model,
        param_grid=param_grid_rf,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1  # Show progress
    )
    grid_rf.fit(X_train_num, y_train)

    best_rf = grid_rf.best_estimator_
    rf_pred = best_rf.predict_proba(X_test_num)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    results['Random Forest'] = rf_auc
    predictions['Random Forest'] = rf_pred
    print(f"Best params: {grid_rf.best_params_}")
    print(f"AUC: {rf_auc:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(best_rf, 'best_random_forest.pkl')

    # ============================================
    # 4. LOGISTIC REGRESSION
    # ============================================
    print("\n" + "="*60)
    print("TUNING LOGISTIC REGRESSION")
    print("="*60)
    start = time.time()

    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    param_grid_lr = {
        'C': [0.1, 1.0, 10.0]
    }

    grid_lr = GridSearchCV(
        lr_model,
        param_grid=param_grid_lr,
        scoring='roc_auc',
        cv=3,
        n_jobs=-1,
        verbose=1  # Show progress
    )
    print("\nFilling NaN for Logistic Regression...")
    X_train_num_filled = X_train_num.fillna(0)
    X_test_num_filled = X_test_num.fillna(0)

    grid_lr.fit(X_train_num_filled, y_train)

    best_lr = grid_lr.best_estimator_
    lr_pred = best_lr.predict_proba(X_test_num_filled)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_pred)
    results['Logistic Regression'] = lr_auc
    predictions['Logistic Regression'] = lr_pred
    print(f"Best params: {grid_lr.best_params_}")
    print(f"AUC: {lr_auc:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(best_lr, 'best_logistic_regression.pkl')

    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("MODEL COMPARISON (TEST AUC)")
    print("="*60)
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<25} AUC: {auc:.4f}")

    # Save results and predictions
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'AUC'])
    results_df = results_df.sort_values('AUC', ascending=False)
    results_df.to_csv('ml_model_comparison.csv', index=False)

    pred_df = pd.DataFrame({'TransactionID': test['TransactionID'].values})
    for name, pred in predictions.items():
        pred_df[name + '_fraud_prob'] = pred
    pred_df.to_csv('ml_model_test_predictions.csv', index=False)

    print("\nResults saved to ml_model_comparison.csv")
    print("Predictions saved to ml_model_test_predictions.csv")
    print("Models saved as pickle files.")

if __name__ == "__main__":
    main()