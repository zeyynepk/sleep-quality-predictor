import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
)

from prepareData import prepareData

# ============================================================
# GENEL AYARLAR
# ============================================================
SEED = 42
SAVE_DIR = Path("modeller")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "svr_regressor_final.joblib"

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def adjusted_r2(r2: float, n: int, p: int) -> float:
    denom = max(1, n - p - 1)
    return 1.0 - (1.0 - r2) * (n - 1) / denom

# ============================================================
# ANA FONKSİYON
# ============================================================
def main():
    print(f"\n[{timestamp()}] SVR (Tamamen 10-Fold) Başlatılıyor...")

    # 1. VERİ HAZIRLIĞI
    features, target_values, preprocessor = prepareData()
    
    target_values = pd.Series(target_values).astype(float).reset_index(drop=True)
    features = features.reset_index(drop=True)
    n_samples = len(target_values)
    
    print(f"✔ Veri: {n_samples} satır.")

    # 2. GRID SEARCH (EN İYİ PARAMETRELERİ BULMA)
    # DÜZELTME: Burası da artık 10-Fold yapıldı.
    print("\n=== SVR — GRID SEARCH (10-FOLD) BAŞLIYOR ===")
    
    param_grid = {
        "reg__C": [1.0, 10.0, 50.0],
        "reg__epsilon": [0.01, 0.1, 0.2, 0.5],
        "reg__gamma": ["scale", "auto"],
        "reg__kernel": ["rbf"], 
    }

    base_pipeline = Pipeline([
        ("pre", clone(preprocessor)),
        ("reg", SVR())
    ])

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring="r2",
        # BURASI ARTIK 10-FOLD
        cv=KFold(n_splits=10, shuffle=True, random_state=SEED), 
        n_jobs=-1, 
        verbose=1
    )

    grid_search.fit(features, target_values)

    best_params = grid_search.best_params_
    print(f"\n✔ En iyi parametreler: {best_params}")
    print(f"✔ En iyi CV R2 Skoru: {grid_search.best_score_:.4f}")

    # 3. 10-FOLD CROSS-VALIDATION (OOF TAHMİN)
    print("\n=== SVR — FİNAL TEST (10-FOLD) BAŞLIYOR ===")
    
    kfold_10 = KFold(n_splits=10, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros(n_samples, dtype=float)
    fold_r2_scores = []

    for fold, (train_idx, valid_idx) in enumerate(kfold_10.split(features), start=1):
        X_train, X_valid = features.iloc[train_idx], features.iloc[valid_idx]
        y_train, y_valid = target_values.iloc[train_idx], target_values.iloc[valid_idx]

        # En iyi parametrelerle modeli kur
        model = SVR(
            C=best_params["reg__C"],
            epsilon=best_params["reg__epsilon"],
            gamma=best_params["reg__gamma"],
            kernel="rbf"
        )
        
        pipeline = Pipeline([
            ("pre", clone(preprocessor)),
            ("reg", model)
        ])

        pipeline.fit(X_train, y_train)
        
        preds = pipeline.predict(X_valid)
        oof_predictions[valid_idx] = preds
        
        r2 = r2_score(y_valid, preds)
        fold_r2_scores.append(r2)
        print(f"[Fold {fold:02d}] R2: {r2:.3f}")

    # 4. SONUÇLAR VE METRİKLER
    mae_oof = mean_absolute_error(target_values, oof_predictions)
    rmse_oof = np.sqrt(mean_squared_error(target_values, oof_predictions))
    r2_oof = r2_score(target_values, oof_predictions)
    evs_oof = explained_variance_score(target_values, oof_predictions)
    
    dummy_pre = clone(preprocessor).fit(features)
    n_features_transformed = dummy_pre.transform(features).shape[1]
    r2_adj_oof = adjusted_r2(r2_oof, n=n_samples, p=n_features_transformed)

    print("-" * 40)
    print("SVR — NİHAİ SONUÇLAR")
    print("-" * 40)
    print(f"MAE      : {mae_oof:.3f}")
    print(f"RMSE     : {rmse_oof:.3f}")
    print(f"R2       : {r2_oof:.3f}")
    print(f"Adj. R2  : {r2_adj_oof:.3f}")
    print(f"EVS      : {evs_oof:.3f}")
    print("-" * 40)

    # 5. GRAFİK
    plt.figure(figsize=(6, 5))
    plt.scatter(target_values, oof_predictions, s=30, edgecolor="black", alpha=0.7, label="Tahminler")
    lo, hi = float(np.min(target_values)), float(np.max(target_values))
    plt.plot([lo, hi], [lo, hi], "r--", lw=2, label="Mükemmel")
    
    plt.title(f"SVR Model Sonuçları (10-Fold Grid + 10-Fold Test)\nR2: {r2_oof:.3f} | RMSE: {rmse_oof:.3f}")
    plt.xlabel("Gerçek Değerler")
    plt.ylabel("Tahmin Edilen Değerler")
    plt.legend()
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"svr_scatter_{timestamp()}.png", dpi=300)
    plt.show()

    # 6. KAYIT
    print("\nModel diske kaydediliyor...")
    final_svr = SVR(
        C=best_params["reg__C"],
        epsilon=best_params["reg__epsilon"],
        gamma=best_params["reg__gamma"],
        kernel="rbf"
    )
    final_pipeline = Pipeline([
        ("pre", clone(preprocessor)),
        ("reg", final_svr)
    ])
    
    final_pipeline.fit(features, target_values)
    joblib.dump(final_pipeline, MODEL_PATH)
    
    print(f"✔ Kayıt Tamam: {MODEL_PATH}")

if __name__ == "__main__":
    main()