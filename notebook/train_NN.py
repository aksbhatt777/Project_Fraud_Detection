# train_NN.py
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

def main():
    print("="*60)
    print("LOADING DATA FOR NEURAL NETWORK")
    print("="*60)
    
    # Load clean data (ensure no NaN)
    df = pd.read_parquet('/home/ubuntu/ann/notebook/fraud_data_clean.parquet')
    df = df.sort_values('TransactionDT').reset_index(drop=True)
    
    # Time-based split: first 80% train, last 20% test
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    target = 'isFraud'
    features = [col for col in df.columns if col not in ['TransactionID', 'isFraud', 'TransactionDT']]
    
    X_train_full = train[features]
    y_train_full = train[target]
    X_test = test[features]
    y_test = test[target]
    
    # Encode categoricals
    cat_cols = X_train_full.select_dtypes(include=['category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(pd.concat([X_train_full[col].astype(str), X_test[col].astype(str)]))
        X_train_full[col] = le.transform(X_train_full[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
    
    # Convert to float32 and fill any remaining NaN with 0
    X_train_full = X_train_full.astype(np.float32).fillna(0)
    X_test = X_test.astype(np.float32).fillna(0)
    
    # Scale features (fit only on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    # Clip extreme values to prevent instability
    X_train_scaled = np.clip(X_train_scaled, -5, 5)
    X_test_scaled = np.clip(X_test_scaled, -5, 5)
    
    # Time-based validation split: last 15% of training data
    # (already sorted by time)
    val_split = int(len(X_train_scaled) * 0.85)
    X_train_nn = X_train_scaled[:val_split]
    y_train_nn = y_train_full.iloc[:val_split].values
    X_val_nn = X_train_scaled[val_split:]
    y_val_nn = y_train_full.iloc[val_split:].values
    
    print(f"Train shape: {X_train_nn.shape}, Validation shape: {X_val_nn.shape}, Test shape: {X_test_scaled.shape}")
    
    # Class weights (capped to avoid extreme weighting)
    scale_pos_weight = (y_train_nn == 0).sum() / (y_train_nn == 1).sum()
    class_weights = {0: 1.0, 1: min(scale_pos_weight, 10.0)}
    print(f"Class weights: {class_weights}")
    
    # Build regularized model (no BatchNorm, more dropout, L2)
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(0.001),
                           input_shape=(X_train_scaled.shape[1],)),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Optimizer with gradient clipping and lower learning rate
    optimizer = keras.optimizers.Adam(
        learning_rate=0.0001,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='auc')]
    )
    
    # Early stopping on validation AUC (time-based validation)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train
    print("\nTraining Neural Network...")
    start = time.time()
    history = model.fit(
        X_train_nn, y_train_nn,
        validation_data=(X_val_nn, y_val_nn),
        epochs=100,              # allow more epochs, early stopping will stop
        batch_size=1024,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1
    )
    
    print(f"\nTraining stopped at epoch: {len(history.history['loss'])}")
    print(f"Best validation AUC: {max(history.history['val_auc']):.4f}")
    
    # Evaluate on test set
    nn_pred = model.predict(X_test_scaled).flatten()
    
    # Check for NaN
    if np.isnan(nn_pred).sum() > 0:
        print("ERROR: NaN in predictions")
        return
    
    nn_auc = roc_auc_score(y_test, nn_pred)
    print(f"\nNeural Network Test AUC: {nn_auc:.4f}")
    print(f"Time: {time.time()-start:.1f}s")
    
    # Save model in native Keras format
    model.save('best_neural_network.keras')
    print("Model saved to best_neural_network.keras")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'TransactionID': test['TransactionID'].values,
        'NN_fraud_prob': nn_pred
    })
    pred_df.to_csv('nn_test_predictions.csv', index=False)
    print("Predictions saved to nn_test_predictions.csv")

if __name__ == "__main__":
    main()