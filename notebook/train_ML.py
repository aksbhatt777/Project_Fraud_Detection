# train_ml_fast.py
import pandas as pd
import numpy as np
import time
import warnings
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("LOADING DATA (FAST MODE - NO GRID SEARCH)")
    print("="*60)
    df = pd.read_parquet('/home/ubuntu/ann/notebook/fraud_data_clean_forML.parquet')
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    target = 'isFraud'
    features = [col for col in df.columns if col not in ['TransactionID', 'isFraud', 'TransactionDT']]
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Prepare numeric version for sklearn models
    cat_cols = X_train.select_dtypes(include=['category']).columns
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
    # 1. LIGHTGBM (Single fit - ~30 sec)
    # ============================================
    print("\nTraining LightGBM...")
    start = time.time()
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=255,
        scale_pos_weight=scale_pos_weight,
        verbosity=-1,
        random_state=42,
        n_jobs=-1
    )
    lgb_model.fit(X_train, y_train, categorical_feature='auto')
    lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
    results['LightGBM'] = roc_auc_score(y_test, lgb_pred)
    predictions['LightGBM'] = lgb_pred
    print(f"  AUC: {results['LightGBM']:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(lgb_model, 'best_lightgbm.pkl')
    
    # ============================================
    # 2. XGBOOST (Single fit - ~1 min)
    # ============================================
    print("\nTraining XGBoost...")
    start = time.time()
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        scale_pos_weight=scale_pos_weight,
        verbosity=0,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train_num, y_train)
    xgb_pred = xgb_model.predict_proba(X_test_num)[:, 1]
    results['XGBoost'] = roc_auc_score(y_test, xgb_pred)
    predictions['XGBoost'] = xgb_pred
    print(f"  AUC: {results['XGBoost']:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(xgb_model, 'best_xgboost.pkl')
    
    # ============================================
    # 3. RANDOM FOREST (Single fit - ~2-3 min)
    # ============================================
    print("\nTraining Random Forest...")
    start = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        max_depth=15,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_num, y_train)
    rf_pred = rf_model.predict_proba(X_test_num)[:, 1]
    results['Random Forest'] = roc_auc_score(y_test, rf_pred)
    predictions['Random Forest'] = rf_pred
    print(f"  AUC: {results['Random Forest']:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(rf_model, 'best_random_forest.pkl')
    
    # ============================================
    # 4. LOGISTIC REGRESSION (Single fit - ~30 sec)
    # ============================================
    print("\nFilling NaN for Logistic Regression...")
    X_train_num_filled = X_train_num.fillna(0)
    X_test_num_filled = X_test_num.fillna(0)

    print("\nTraining Logistic Regression...")
    start = time.time()
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    lr_model.fit(X_train_num_filled, y_train)
    lr_pred = lr_model.predict_proba(X_test_num_filled)[:, 1]
    results['Logistic Regression'] = roc_auc_score(y_test, lr_pred)
    predictions['Logistic Regression'] = lr_pred
    print(f"  AUC: {results['Logistic Regression']:.4f} | Time: {time.time()-start:.1f}s")
    joblib.dump(lr_model, 'best_logistic_regression.pkl')
    
    # ============================================
    # SUMMARY
    # ============================================
    print("\n" + "="*60)
    print("MODEL COMPARISON (TEST AUC)")
    print("="*60)
    for name, auc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:<25} AUC: {auc:.4f}")
    
    # Save results
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'AUC'])
    results_df = results_df.sort_values('AUC', ascending=False)
    results_df.to_csv('ml_model_comparison.csv', index=False)
    
    pred_df = pd.DataFrame({'TransactionID': test['TransactionID'].values})
    for name, pred in predictions.items():
        pred_df[name + '_fraud_prob'] = pred
    pred_df.to_csv('ml_model_test_predictions.csv', index=False)
    
    print("\nResults saved!")

if __name__ == "__main__":
    main()
