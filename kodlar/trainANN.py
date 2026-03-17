import os
# ============================================================
# 1. KİLİT MEKANİZMASI (SÜPER KİLİT)
# ============================================================
# Bu ayarlar TensorFlow import edilmeden ÖNCE yapılmalıdır!

# a) oneDNN optimizasyonunu kapat (Sonucun değişmesine neden olabilir)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# b) TensorFlow'un gereksiz tüm uyarılarını (Kırmızı yazıları) kapat
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# c) GPU'yu devre dışı bırak (Sıralı işlem için şart)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# d) Rastgelelik kaynaklarını kilitle
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random
import numpy as np
import tensorflow as tf

# Kütüphane seviyesinde seed ayarları
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Kalan kütüphaneler
import logging
import time
import itertools

# Python loglarını da sustur
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)
from sklearn.base import clone
from scipy import sparse

# Senin veri hazırlama dosyan
from prepareData import prepareData 
from tensorflow import keras

# ============================================================
# 2. AYARLAR
# ============================================================
SAVE_DIR = Path("modeller")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "ann_regressor_fixed.h5"

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n - p - 1 <= 0: return r2
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def to_dense(array_like):
    return array_like.toarray() if sparse.issparse(array_like) else np.asarray(array_like)

# ============================================================
# 3. MODEL MİMARİSİ
# ============================================================
def build_ann(n_features, units=64, learning_rate=0.01, dropout=0.2):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(units, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(max(units // 2, 1), activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(1)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model

# ============================================================
# 4. MAIN
# ============================================================
def main():
    print(f"\n[{timestamp()}] ANN (TAM KİLİTLİ MOD) Başlatılıyor...")
    print("Not: Bu modda sonuçlar asla değişmez.")

    # A) VERİ YÜKLEME
    features, target_values, preprocessor = prepareData()
    target_values = pd.Series(target_values).astype(float).reset_index(drop=True)
    features = features.reset_index(drop=True)
    n_samples = len(target_values)
    
    print(f"✔ Veri: {n_samples} satır.")

    # B) GRID SEARCH
    print("\n=== ADIM 1: OPTİMİZASYON (GRID SEARCH) ===")
    param_grid = {
        "units":         [64, 128],
        "learning_rate": [0.01, 0.001],
        "dropout":       [0.2, 0.3]
    }
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -np.inf
    best_params = None
    
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=SEED)
    
    for i, params in enumerate(combinations, 1):
        fold_r2s = []
        for train_idx, val_idx in cv_strategy.split(features):
            X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_tr, y_val = target_values.iloc[train_idx], target_values.iloc[val_idx]
            
            fold_pre = clone(preprocessor)
            X_tr_sc = to_dense(fold_pre.fit_transform(X_tr))
            X_val_sc = to_dense(fold_pre.transform(X_val))
            
            model = build_ann(X_tr_sc.shape[1], **params)
            
            cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            # verbose=0: Ekrana eğitim satırlarını yazma (temiz çıktı için)
            model.fit(X_tr_sc, y_tr, validation_data=(X_val_sc, y_val),
                      epochs=60, batch_size=32, verbose=0, callbacks=cb)
            
            pred = model.predict(X_val_sc, verbose=0).flatten()
            fold_r2s.append(r2_score(y_val, pred))
            
        avg_r2 = np.mean(fold_r2s)
        print(f"   [{i}/{len(combinations)}] Ayarlar: {params} -> R2: {avg_r2:.3f}")
        
        if avg_r2 > best_score:
            best_score = avg_r2
            best_params = params

    print(f"\n✔ EN İYİ AYARLAR: {best_params}")

    # C) FİNAL TEST
    print("\n=== ADIM 2: DETAYLI PERFORMANS ANALİZİ ===")
    oof_predictions = np.zeros(n_samples)
    first_fold_history = None
    
    for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(features), start=1):
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = target_values.iloc[train_idx], target_values.iloc[valid_idx]
        
        fold_pre = clone(preprocessor)
        X_train_sc = to_dense(fold_pre.fit_transform(X_train))
        X_valid_sc = to_dense(fold_pre.transform(X_valid))
        
        model = build_ann(X_train_sc.shape[1], **best_params)
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        history = model.fit(
            X_train_sc, y_train,
            validation_data=(X_valid_sc, y_valid),
            epochs=150, batch_size=32, verbose=0, callbacks=callbacks
        )
        
        if fold == 1: first_fold_history = history
        oof_predictions[valid_idx] = model.predict(X_valid_sc, verbose=0).flatten()

    # D) SONUÇLAR
    mae_oof  = mean_absolute_error(target_values, oof_predictions)
    rmse_oof = np.sqrt(mean_squared_error(target_values, oof_predictions))
    r2_oof   = r2_score(target_values, oof_predictions)
    evs_oof  = explained_variance_score(target_values, oof_predictions)
    r2_adj_oof = adjusted_r2(r2_oof, n=n_samples, p=X_train_sc.shape[1])

    print("-" * 40)
    print("ANN REGRESSOR — NİHAİ SABİT SONUÇLAR")
    print("-" * 40)
    print(f"MAE      : {mae_oof:.3f}")
    print(f"RMSE     : {rmse_oof:.3f}")
    print(f"R2       : {r2_oof:.3f}")
    print(f"Adj. R2  : {r2_adj_oof:.3f}")
    print(f"EVS      : {evs_oof:.3f}")
    print("-" * 40)

    # E) GRAFİKLER
    plt.figure(figsize=(6, 5))
    plt.scatter(target_values, oof_predictions, s=40, edgecolor="black", alpha=0.7)
    low, high = float(min(target_values.min(), oof_predictions.min())), float(max(target_values.max(), oof_predictions.max()))
    plt.plot([low, high], [low, high], "r--", lw=2)
    plt.title(f"ANN Regressor\nR2: {r2_oof:.3f} | EVS: {evs_oof:.3f}")
    plt.xlabel("Gerçek"); plt.ylabel("Tahmin")
    plt.tight_layout(); plt.savefig(SAVE_DIR / f"ann_scatter_{timestamp()}.png", dpi=400); plt.show()

    residuals = target_values - oof_predictions
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black", color="green", alpha=0.7)
    plt.title("Hata Dağılımı")
    plt.tight_layout(); plt.savefig(SAVE_DIR / f"ann_residuals_{timestamp()}.png", dpi=400); plt.show()
    
    if first_fold_history:
        plt.figure(figsize=(6, 5))
        plt.plot(first_fold_history.history['loss'], label='Train')
        plt.plot(first_fold_history.history['val_loss'], label='Val')
        plt.title('Loss Curve'); plt.legend()
        plt.tight_layout(); plt.savefig(SAVE_DIR / f"ann_loss_curve_{timestamp()}.png", dpi=400); plt.show()

    # F) KAYIT
    print("\nModel kaydediliyor...")
    final_pre = clone(preprocessor)
    X_full = to_dense(final_pre.fit_transform(features))
    final_model = build_ann(X_full.shape[1], **best_params)
    final_model.fit(X_full, target_values, epochs=150, batch_size=32, verbose=0)
    final_model.save(MODEL_PATH)
    print(f"✔ İşlem Tamam. Kayıt: {MODEL_PATH}")

if __name__ == "__main__":
    main()