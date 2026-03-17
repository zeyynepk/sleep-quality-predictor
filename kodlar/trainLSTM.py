import os

# ============================================================
# 1. KARARLI KİLİT MEKANİZMASI (GÜVENLİ & SABİT MOD)
# ============================================================
# Bu ayarlar TensorFlow import edilmeden ÖNCE yapılmalıdır.

# 1. GPU'yu kapat (Sonucun değişmemesi için en kritik ayar budur)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 2. İşlemci Hızlandırmalarını (oneDNN) kapat
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 3. Python Hash Seed'i kilitle
os.environ['PYTHONHASHSEED'] = '42'

# 4. Deterministik Operasyonları Zorla
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import random
import time
import logging
import itertools
from pathlib import Path
import joblib

# Gerekli kütüphaneler
import numpy as np
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

import tensorflow as tf

# Kütüphane seviyesinde seed ayarları
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# TensorFlow uyarılarını sustur
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Thread ayarını OS seviyesinde değil, TF seviyesinde yapıyoruz (Daha güvenli)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Sarı çizgi uyarısını görmezden gelmek için type: ignore ekli
from tensorflow.keras import layers, callbacks, optimizers, losses, models, initializers # type: ignore

from prepareData import prepareData

# ============================================================
# 2. AYARLAR
# ============================================================
SAVE_DIR = Path("modeller")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "lstm_regressor_final.keras"
PREPROC_PATH = SAVE_DIR / "lstm_preprocessor_final.joblib"

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def adjusted_r2(r2: float, n: int, p: int) -> float:
    denom = max(1, n - p - 1)
    return 1.0 - (1.0 - r2) * (n - 1) / denom

def to_3d(x2d: np.ndarray) -> np.ndarray:
    """
    LSTM 3 boyutlu veri ister: (Samples, TimeSteps, Features)
    Burada TimeSteps=1 olarak kabul ediyoruz.
    """
    return x2d.reshape((x2d.shape[0], 1, x2d.shape[1]))

# ============================================================
# 3. LSTM MODEL MİMARİSİ (DİNAMİK PARAMETRELİ)
# ============================================================
def build_lstm(n_features: int, units=32, learning_rate=0.001) -> tf.keras.Model:
    inp = layers.Input(shape=(1, n_features))
    
    # LSTM Katmanı (Initializer kilitli)
    x = layers.LSTM(
        units, 
        return_sequences=False,
        kernel_initializer=initializers.GlorotUniform(seed=SEED),
        recurrent_initializer=initializers.Orthogonal(seed=SEED),
        bias_initializer='zeros'
    )(inp)
    
    x = layers.Dense(
        units, 
        activation="relu",
        kernel_initializer=initializers.GlorotUniform(seed=SEED)
    )(x)
    
    out = layers.Dense(
        1, 
        activation="linear",
        kernel_initializer=initializers.GlorotUniform(seed=SEED)
    )(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.MeanSquaredError(),
        metrics=["mae"]
    )
    return model

# ============================================================
# 4. MAIN
# ============================================================
def main():
    print(f"\n[{timestamp()}] LSTM (GRID SEARCH + SABİT MOD) Başlatılıyor...")
    print("Not: İşlemci tek çekirdeğe kilitlendiği için bu işlem biraz sürebilir.")

    # A) VERİ YÜKLEME
    features, target_values, preprocessor = prepareData()
    target_values = pd.Series(target_values).astype(float).reset_index(drop=True)
    features = features.reset_index(drop=True)
    n_samples = len(target_values)
    
    print(f"✔ Veri: {n_samples} satır.")

    # B) GRID SEARCH (HİPERPARAMETRE ARAMA)
    print("\n=== ADIM 1: OPTİMİZASYON (GRID SEARCH) ===")
    
    # LSTM için en kritik 2 parametreyi seçtik (Toplam 4 kombinasyon)
    # Çok fazla artırırsak süre çok uzar.
    param_grid = {
        "units":         [32, 64],
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
        # Her kombinasyon için 10-Fold CV yapıyoruz
        for train_idx, val_idx in cv_strategy.split(features):
            X_tr, X_val = features.iloc[train_idx], features.iloc[val_idx]
            y_tr, y_val = target_values.iloc[train_idx], target_values.iloc[val_idx]
            
            # Preprocessing
            fold_pre = clone(preprocessor)
            X_tr_sc = fold_pre.fit_transform(X_tr)
            X_val_sc = fold_pre.transform(X_val)
            
            # 3D Reshape
            X_tr_3d = to_3d(np.asarray(X_tr_sc))
            X_val_3d = to_3d(np.asarray(X_val_sc))
            
            # Model Kur
            model = build_lstm(
                n_features=X_tr_3d.shape[-1],
                units=params["units"],
                learning_rate=params["learning_rate"]
            )
            
            # Grid Search sırasında epoch sayısını biraz düşük tutuyoruz (Hız için)
            cb = [callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
            model.fit(
                X_tr_3d, y_tr, 
                validation_data=(X_val_3d, y_val),
                epochs=40,  # Hızlı arama için yeterli
                batch_size=32, 
                verbose=0, 
                callbacks=cb,
                shuffle=False # Deterministik olması için shuffle kapalı
            )
            
            pred = model.predict(X_val_3d, verbose=0).ravel()
            fold_r2s.append(r2_score(y_val, pred))
            
        avg_r2 = np.mean(fold_r2s)
        print(f"   [{i}/{len(combinations)}] Ayarlar: {params} -> R2: {avg_r2:.3f}")
        
        if avg_r2 > best_score:
            best_score = avg_r2
            best_params = params

    print(f"\n✔ EN İYİ AYARLAR: {best_params}")

    # C) FİNAL TEST (EN İYİ AYARLARLA DETAYLI ANALİZ)
    print("\n=== ADIM 2: DETAYLI PERFORMANS ANALİZİ (10-FOLD ORTALAMA) ===")
    oof_predictions = np.zeros(n_samples)
    
    # Tüm foldların loss değerlerini saklamak için listeler
    all_train_losses = []
    all_val_losses = []

    for fold, (train_idx, valid_idx) in enumerate(cv_strategy.split(features), start=1):
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = target_values.iloc[train_idx], target_values.iloc[valid_idx]
        
        fold_pre = clone(preprocessor)
        X_train_t = fold_pre.fit_transform(X_train)
        X_valid_t = fold_pre.transform(X_valid)

        X_train_3d = to_3d(np.asarray(X_train_t))
        X_valid_3d = to_3d(np.asarray(X_valid_t))
        
        # En iyi parametrelerle modeli kur
        model = build_lstm(
            n_features=X_train_3d.shape[-1],
            units=best_params["units"],
            learning_rate=best_params["learning_rate"]
        )

        es = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        
        # History'yi alıyoruz
        history = model.fit(
            X_train_3d, y_train,
            validation_data=(X_valid_3d, y_valid),
            epochs=150, 
            batch_size=32,
            verbose=0, 
            callbacks=[es],
            shuffle=False
        )
        
        # Bu foldun loss değerlerini listeye ekle
        all_train_losses.append(history.history['loss'])
        all_val_losses.append(history.history['val_loss'])

        oof_predictions[valid_idx] = model.predict(X_valid_3d, verbose=0).ravel()
        print(f"Fold {fold} tamamlandı. (Epoch: {len(history.history['loss'])})")

    # --- ORTALAMA HESAPLAMA MANTIĞI ---
    # Her fold farklı epochta bittiği için en uzuna göre eşitleme (padding) yapıyoruz.
    # Kısa bitenlerin son değerini uzatıyoruz ki ortalama bozulmasın.
    max_len = max(len(h) for h in all_train_losses)

    def pad_loss(loss_list, target_len):
        # Son değeri alıp kalan boşlukları onla doldurur
        return loss_list + [loss_list[-1]] * (target_len - len(loss_list))

    padded_train = np.array([pad_loss(h, max_len) for h in all_train_losses])
    padded_val = np.array([pad_loss(h, max_len) for h in all_val_losses])

    # Ortalamalar ve Standart Sapmalar
    mean_train = np.mean(padded_train, axis=0)
    std_train = np.std(padded_train, axis=0)
    
    mean_val = np.mean(padded_val, axis=0)
    std_val = np.std(padded_val, axis=0)

    # D) SONUÇLAR
    mae_oof  = mean_absolute_error(target_values, oof_predictions)
    rmse_oof = np.sqrt(mean_squared_error(target_values, oof_predictions))
    r2_oof   = r2_score(target_values, oof_predictions)
    evs_oof  = explained_variance_score(target_values, oof_predictions)
    r2_adj_oof = adjusted_r2(r2_oof, n=n_samples, p=X_train_t.shape[1])

    print("-" * 40)
    print("LSTM REGRESSOR — NİHAİ 10-FOLD CV SONUÇLARI")
    print("-" * 40)
    print(f"MAE      : {mae_oof:.3f}")
    print(f"RMSE     : {rmse_oof:.3f}")
    print(f"R2       : {r2_oof:.3f}")
    print(f"Adj. R2  : {r2_adj_oof:.3f}")
    print("-" * 40)

    # ============================================================
    # E) GRAFİKLER (GÜNCELLENDİ - ORTALAMA LOSS)
    # ============================================================
    
    # 1. GRAFİK: Gerçek vs Tahmin (Scatter)
    plt.figure(figsize=(6, 5))
    plt.scatter(target_values, oof_predictions, s=30, edgecolor="black", alpha=0.7)
    lo, hi = float(np.min(target_values)), float(np.max(target_values))
    plt.plot([lo, hi], [lo, hi], "r--", lw=2)
    plt.title(f"LSTM Regressor (10-Fold CV)\nR2: {r2_oof:.3f}")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Değerleri")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"lstm_grid_scatter_{timestamp()}.png", dpi=400)
    plt.close()

    # 2. GRAFİK: 10 FOLD ORTALAMA LOSS CURVE (GÖLGELİ)
    plt.figure(figsize=(10, 6))
    epochs = range(1, max_len + 1)

    # Eğitim (Train) Çizgisi ve Gölgesi
    plt.plot(epochs, mean_train, 'b-', label='Ortalama Eğitim Kaybı (Train Loss)', linewidth=2)
    plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, color='b', alpha=0.15)

    # Doğrulama (Validation) Çizgisi ve Gölgesi
    plt.plot(epochs, mean_val, 'r-', label='Ortalama Doğrulama Kaybı (Val Loss)', linewidth=2)
    plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, color='r', alpha=0.15)

    plt.title("LSTM Model Eğitim Dinamikleri (10-Fold)", fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    save_path = SAVE_DIR / f"lstm_avg_loss_curve_{timestamp()}.png"
    plt.savefig(save_path, dpi=400)
    print(f"✔ 10-Fold Ortalama Loss grafiği kaydedildi: {save_path}")
    plt.show()

    # F) KAYIT
    print("\nModel kaydediliyor...")
    final_pre = clone(preprocessor)
    X_all = final_pre.fit_transform(features)
    X_all_3d = to_3d(np.asarray(X_all))
    
    final_model = build_lstm(
        n_features=X_all_3d.shape[-1],
        units=best_params["units"],
        learning_rate=best_params["learning_rate"]
    )
    final_model.fit(
        X_all_3d, target_values.values, 
        epochs=150, batch_size=32, verbose=0, shuffle=False
    )
    final_model.save(MODEL_PATH)
    joblib.dump(final_pre, PREPROC_PATH)
    print("✔ İşlem Tamam.")

if __name__ == "__main__":
    main()