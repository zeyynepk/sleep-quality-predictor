import os

# ============================================================
# 1. KİLİT MEKANİZMASI (SÜPER KİLİT - CNN İÇİN)
# ============================================================
# Bu ayarlar TensorFlow import edilmeden ÖNCE yapılmalıdır!

# a) oneDNN optimizasyonunu kapat
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# b) TensorFlow uyarılarını sustur
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# c) GPU'yu devre dışı bırak (Sonuç sabitlemek için şart)
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

logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
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
MODEL_PATH = SAVE_DIR / "cnn_regressor_fixed.keras"

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def adjusted_r2(r2: float, n: int, p: int) -> float:
    if n - p - 1 <= 0: return r2
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def to_dense(array_like):
    return array_like.toarray() if sparse.issparse(array_like) else np.asarray(array_like)

def to_sequence(features_2d: np.ndarray):
    """
    CNN İÇİN ÖZEL FONKSİYON:
    Veriyi (Samples, Features) formatından (Samples, Features, 1) formatına çevirir.
    Çünkü Conv1D katmanı 3 boyutlu veri ister.
    """
    features_2d = np.asarray(features_2d, dtype=np.float32)
    return features_2d.reshape((features_2d.shape[0], features_2d.shape[1], 1))

# ============================================================
# 3. CNN MODEL MİMARİSİ (DİNAMİK PARAMETRELİ)
# ============================================================
def build_cnn(n_steps, filters=64, kernel_size=3, learning_rate=0.001):
    """
    Grid Search ile 'filters' ve 'kernel_size' parametreleri denenecek.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(n_steps, 1)),
        
        # 1. Konvolüsyon Bloğu
        keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Dropout(0.25),
        
        # 2. Konvolüsyon Bloğu (Filtre sayısı genelde 2 katına çıkar)
        keras.layers.Conv1D(filters=filters*2, kernel_size=kernel_size, activation="relu", padding="same"),
        keras.layers.Dropout(0.25),
        
        keras.layers.Flatten(),
        
        # Tam Bağlantılı Katman
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1) # Regresyon çıktısı
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
    print(f"\n[{timestamp()}] CNN (TAM KİLİTLİ MOD) Başlatılıyor...")
    print("Not: Bu modda sonuçlar sabittir (Deterministic).")

    # A) VERİ YÜKLEME
    features, target_values, preprocessor = prepareData()
    target_values = pd.Series(target_values).astype(float).reset_index(drop=True)
    features = features.reset_index(drop=True)
    n_samples = len(target_values)
    
    print(f"✔ Veri: {n_samples} satır.")

    # B) GRID SEARCH (HİPERPARAMETRE ARAMA)
    print("\n=== ADIM 1: OPTİMİZASYON (GRID SEARCH) ===")
    
    # CNN için denenecek parametreler
    param_grid = {
        "filters":       [32, 64],
        "kernel_size":   [3, 5],
        "learning_rate": [0.01, 0.001]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_score = -np.inf
    best_params = None
    
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=SEED)
    
    print(f"Toplam {len(combinations)} kombinasyon test ediliyor...")

    for i, params in enumerate(combinations, 1):
        fold_r2s = []
        for train_idx, val_idx in cv_strategy.split(features):
            X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_tr, y_val = target_values.iloc[train_idx], target_values.iloc[val_idx]
            
            # Preprocessing
            fold_pre = clone(preprocessor)
            X_tr_sc = to_dense(fold_pre.fit_transform(X_tr))
            X_val_sc = to_dense(fold_pre.transform(X_val))
            
            # CNN Reshape (3 Boyuta Çevir) - ÖNEMLİ!
            X_tr_seq = to_sequence(X_tr_sc)
            X_val_seq = to_sequence(X_val_sc)
            
            # Model Kur
            model = build_cnn(
                n_steps=X_tr_seq.shape[1],
                filters=params["filters"],
                kernel_size=params["kernel_size"],
                learning_rate=params["learning_rate"]
            )
            
            # Hızlı eğitim
            cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            model.fit(X_tr_seq, y_tr, validation_data=(X_val_seq, y_val),
                      epochs=50, batch_size=32, verbose=0, callbacks=cb)
            
            pred = model.predict(X_val_seq, verbose=0).flatten()
            fold_r2s.append(r2_score(y_val, pred))
            
        avg_r2 = np.mean(fold_r2s)
        print(f"   [{i}/{len(combinations)}] Ayarlar: {params} -> R2: {avg_r2:.3f}")
        
        if avg_r2 > best_score:
            best_score = avg_r2
            best_params = params

    print(f"\n✔ EN İYİ AYARLAR: {best_params}")

    # C) FİNAL TEST (EN İYİ AYARLARLA)
    print("\n=== ADIM 2: DETAYLI PERFORMANS ANALİZİ ===")
    oof_predictions = np.zeros(n_samples)
    first_fold_history = None
    
    for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(features), start=1):
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = target_values.iloc[train_idx], target_values.iloc[valid_idx]
        
        # Preprocessing
        fold_pre = clone(preprocessor)
        X_train_t = to_dense(fold_pre.fit_transform(X_train))
        X_valid_t = to_dense(fold_pre.transform(X_valid))
        
        # CNN Reshape
        X_train_seq = to_sequence(X_train_t)
        X_valid_seq = to_sequence(X_valid_t)
        
        # Best Model
        model = build_cnn(
            n_steps=X_train_seq.shape[1],
            filters=best_params["filters"],
            kernel_size=best_params["kernel_size"],
            learning_rate=best_params["learning_rate"]
        )
        
        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        history = model.fit(
            X_train_seq, y_train,
            validation_data=(X_valid_seq, y_valid),
            epochs=150, batch_size=32, verbose=0, callbacks=callbacks
        )
        
        if fold == 1: first_fold_history = history
        oof_predictions[valid_idx] = model.predict(X_valid_seq, verbose=0).flatten()

    # D) SONUÇLAR
    mae_oof  = mean_absolute_error(target_values, oof_predictions)
    rmse_oof = np.sqrt(mean_squared_error(target_values, oof_predictions))
    r2_oof   = r2_score(target_values, oof_predictions)
    evs_oof  = explained_variance_score(target_values, oof_predictions)
    
    p_features = X_train_t.shape[1]
    r2_adj_oof = adjusted_r2(r2_oof, n=n_samples, p=p_features)

    print("-" * 40)
    print("CNN REGRESSOR — NİHAİ SABİT SONUÇLAR")
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
    plt.title(f"CNN Regressor\nR2: {r2_oof:.3f} | EVS: {evs_oof:.3f}")
    plt.xlabel("Gerçek"); plt.ylabel("Tahmin")
    plt.tight_layout(); plt.savefig(SAVE_DIR / f"cnn_scatter_{timestamp()}.png", dpi=400); plt.show()

    residuals = target_values - oof_predictions
    plt.figure(figsize=(6, 5))
    plt.hist(residuals, bins=20, edgecolor="black", color="orange", alpha=0.7)
    plt.title("Hata Dağılımı")
    plt.tight_layout(); plt.savefig(SAVE_DIR / f"cnn_residuals_{timestamp()}.png", dpi=400); plt.show()
    
    if first_fold_history:
        plt.figure(figsize=(6, 5))
        plt.plot(first_fold_history.history['loss'], label='Train')
        plt.plot(first_fold_history.history['val_loss'], label='Val')
        plt.title('Loss Curve'); plt.legend()
        plt.tight_layout(); plt.savefig(SAVE_DIR / f"cnn_loss_curve_{timestamp()}.png", dpi=400); plt.show()

    # F) KAYIT
    print("\nModel kaydediliyor...")
    final_pre = clone(preprocessor)
    X_full = to_dense(final_pre.fit_transform(features))
    X_full_seq = to_sequence(X_full)
    
    final_model = build_cnn(
        n_steps=X_full_seq.shape[1],
        filters=best_params["filters"],
        kernel_size=best_params["kernel_size"],
        learning_rate=best_params["learning_rate"]
    )
    final_model.fit(X_full_seq, target_values, epochs=150, batch_size=32, verbose=0)
    final_model.save(MODEL_PATH)
    print(f"✔ İşlem Tamam. Kayıt: {MODEL_PATH}")

if __name__ == "__main__":
    main()