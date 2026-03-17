import os, logging, time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    explained_variance_score
)
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

from prepareData import prepareData

# --- AYARLAR ---
SEED = 42
SAVE_DIR = Path("modeller")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "knn_regressor_final.joblib"

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def calculated_adjusted_r2(r2, n, p):
    """
    Düzeltilmiş R-Kare hesaplar.
    n: Örneklem sayısı
    p: Değişken sayısı
    """
    if n - p - 1 <= 0:
        return r2
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def main():
    # --------------------------------------------------------
    # 1) VERI HAZIRLIĞI
    # --------------------------------------------------------
    print("Veri hazırlanıyor...")
    features, target, preprocessor = prepareData()
    
    target = pd.Series(target).astype(float).reset_index(drop=True)
    features = features.reset_index(drop=True)
    
    n_samples = len(target)
    print(f"Veri Seti: {n_samples} satır")

    # --------------------------------------------------------
    # 2) MODEL VE PIPELINE (KNN)
    # --------------------------------------------------------
    # KNN mesafeye dayalı olduğu için ölçekleme (scaling) çok önemlidir.
    # prepareData içindeki preprocessor'ın bunu yaptığını varsayıyoruz.
    knn_base = KNeighborsRegressor()

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("reg", knn_base)
    ])

    # --------------------------------------------------------
    # 3) HİPERPARAMETRE IZGARASI
    # --------------------------------------------------------
    param_grid = {
        "reg__n_neighbors": [3,5,7,9,11],
        "reg__weights": ["uniform"],
        "reg__metric": ["minkowski", "euclidean", "manhattan"],
    }

    # --------------------------------------------------------
    # 4) OPTİMİZASYON: 10-FOLD GRID SEARCH
    # --------------------------------------------------------
    print("\n=== KNN — 10-FOLD GRID SEARCH BAŞLIYOR ===")
    
    # Standartlaştırma: Aynı random_state ile aynı katlamaları (folds) kullanıyoruz.
    cv_strategy = KFold(n_splits=10, shuffle=True, random_state=SEED)
    
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="r2",
        cv=cv_strategy,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(features, target)

    best_model = grid.best_estimator_
    print(f"\n✔ En İyi Parametreler: {grid.best_params_}")
    print(f"✔ Grid Search R2 Skoru: {grid.best_score_:.3f}")

    # --------------------------------------------------------
    # 5) OOF TAHMİNLERİ VE DETAYLI METRİKLER
    # --------------------------------------------------------
    print("\nDetaylı metrikler hesaplanıyor...")
    oof_preds = cross_val_predict(
        best_model, 
        features, 
        target, 
        cv=cv_strategy, 
        n_jobs=-1
    )

    # Temel Metrikler
    mae = mean_absolute_error(target, oof_preds)
    rmse = np.sqrt(mean_squared_error(target, oof_preds))
    r2 = r2_score(target, oof_preds)
    evs = explained_variance_score(target, oof_preds)

    # Adjusted R2 İçin Özellik Sayısını (p) Bulma
    features_transformed = best_model.named_steps["pre"].transform(features)
    p_features = features_transformed.shape[1]
    
    adj_r2 = calculated_adjusted_r2(r2, n_samples, p_features)

    print("\n=== NİHAİ MODEL PERFORMANSI (KNN - 10-Fold CV) ===")
    print(f"MAE      : {mae:.3f}")
    print(f"RMSE     : {rmse:.3f}")
    print(f"R2       : {r2:.3f}")
    print(f"Adj. R2  : {adj_r2:.3f}")
    print(f"EVS      : {evs:.3f}")

    # --------------------------------------------------------
    # 6) GRAFİKLER
    # --------------------------------------------------------
    current_time = timestamp()

    # Grafik 1: Scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(target, oof_preds, s=40, edgecolor="black", alpha=0.75, label="Veri Noktaları")
    low = min(float(target.min()), float(oof_preds.min()))
    high = max(float(target.max()), float(oof_preds.max()))
    plt.plot([low, high], [low, high], color="red", linestyle="--", linewidth=2, label="İdeal")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Değerleri")
    plt.title(f"KNN Regressor (10-Fold CV)\nR2={r2:.3f} | Adj.R2={adj_r2:.3f} | RMSE={rmse:.3f}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"knn_scatter_{current_time}.png", dpi=400)
    plt.show()

    # Grafik 2: Residuals
    residuals = target - oof_preds
    plt.figure(figsize=(6, 5))
    plt.scatter(oof_preds, residuals, s=40, edgecolor="black", alpha=0.75, color="orange")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Tahmin Edilen")
    plt.ylabel("Hata (Gerçek - Tahmin)")
    plt.title(f"Hata Dağılımı (Residuals)\nMAE={mae:.3f} | EVS={evs:.3f}")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"knn_residuals_{current_time}.png", dpi=400)
    plt.show()

    # --------------------------------------------------------
    # 7) KAYIT
    # --------------------------------------------------------
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel kaydedildi: {MODEL_PATH}")

if __name__ == "__main__":
    main()