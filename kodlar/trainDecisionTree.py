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
import seaborn as sns # Renk paleti için eklendi

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    explained_variance_score
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from prepareData import prepareData

# --- SETTINGS ---
SEED = 42
SAVE_DIR = Path("modeller")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = SAVE_DIR / "dt_regressor_final.joblib"

def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def calculated_adjusted_r2(r2, n, p):
    """
    Calculates Adjusted R-Squared.
    n: Number of samples
    p: Number of predictors
    """
    if n - p - 1 <= 0:
        return r2
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)

def main():
    # 1) DATA PREPARATION
    print("Preparing Data...")
    features, target, preprocessor = prepareData()
    
    target = pd.Series(target).astype(float).reset_index(drop=True)
    features = features.reset_index(drop=True)
    
    n_samples = len(target)
    print(f"Dataset Size: {n_samples} rows")

    # 2) MODEL PIPELINE SETUP
    dt_base = DecisionTreeRegressor(
        criterion="squared_error",
        random_state=SEED
    )

    pipeline = Pipeline([
        ("pre", preprocessor),
        ("reg", dt_base)
    ])

    # 3) PARAMETER GRID
    param_grid = {
        "reg__max_depth": [5, 10, None],
        "reg__min_samples_split": [2, 5, 10],
        "reg__min_samples_leaf": [1, 2, 4],
    }

    # 4) GRID SEARCH (Finding Best Params)
    print("\n=== STARTING 10-FOLD GRID SEARCH ===")
    
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
    print(f"\n✔ Best Parameters: {grid.best_params_}")
    print(f"✔ Best Grid R2 Score: {grid.best_score_:.3f}")

    # 5) MANUAL 10-FOLD LOOP FOR PLOTTING & METRICS
    # We do this manually instead of cross_val_predict to handle colors/shapes per fold
    print("\nRunning Manual 10-Fold for Detailed Visualization...")
    
    # Visualization Setup
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    markers = ['o', 's', '^', 'v', '<', '>', 'P', 'X', 'D', '*'] 
    colors = sns.color_palette("tab10", 10)
    
    # Storage for global metrics
    all_y_true = []
    all_y_pred = []
    
    # Loop
    for fold_idx, (train_index, test_index) in enumerate(cv_strategy.split(features, target)):
        # Split Data
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        # Train Best Model on this fold
        best_model.fit(X_train, y_train)
        
        # Predict
        fold_preds = best_model.predict(X_test)
        
        # Store for metrics
        all_y_true.extend(y_test)
        all_y_pred.extend(fold_preds)
        
        # --- JITTER & PLOTTING ---
        # Add slight random noise for visualization only (does not affect metrics)
        jitter_x = np.random.normal(0, 0.08, size=len(y_test))
        jitter_y = np.random.normal(0, 0.08, size=len(fold_preds))
        
        plt.scatter(
            y_test + jitter_x, 
            fold_preds + jitter_y,
            label=f'Fold {fold_idx + 1}',
            color=colors[fold_idx],
            marker=markers[fold_idx],
            s=70,
            alpha=0.75,
            edgecolor='black',
            linewidth=0.5
        )

    # Convert lists to arrays for metric calculation
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # 6) CALCULATE METRICS
    mae = mean_absolute_error(all_y_true, all_y_pred)
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    r2 = r2_score(all_y_true, all_y_pred)
    evs = explained_variance_score(all_y_true, all_y_pred)

    # Adjusted R2 Calculation
    # Transform features to get column count (p)
    features_transformed = best_model.named_steps["pre"].transform(features)
    p_features = features_transformed.shape[1]
    adj_r2 = calculated_adjusted_r2(r2, n_samples, p_features)

    print("\n=== FINAL MODEL PERFORMANCE (10-Fold CV) ===")
    print(f"MAE      : {mae:.3f}")
    print(f"RMSE     : {rmse:.3f}")
    print(f"R2       : {r2:.3f}")
    print(f"Adj. R2  : {adj_r2:.3f}")
    print(f"EVS      : {evs:.3f}")

    # 7) FINALIZE SCATTER PLOT (English)
    current_time = timestamp()
    
    # Ideal Line
    global_min = min(min(all_y_true), min(all_y_pred))
    global_max = max(max(all_y_true), max(all_y_pred))
    plt.plot([global_min, global_max], [global_min, global_max], 
             color='red', linestyle='--', linewidth=3, label='Ideal Line (y=x)')

    plt.title(f"Decision Tree - Actual vs Predicted (10-Fold CV)\n$R^2$ = {r2:.3f} | RMSE = {rmse:.3f}", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Actual Quality of Sleep Score", fontsize=14)
    plt.ylabel("Predicted Quality of Sleep Score", fontsize=14)
    
    # Legend outside
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Iterations")
    
    # Ticks adjustment
    plt.xticks(np.arange(int(global_min), int(global_max)+1, 1))
    plt.yticks(np.arange(int(global_min), int(global_max)+1, 1))
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"dt_scatter_english_{current_time}.png", dpi=300, bbox_inches='tight')
    plt.show()

   # 8) COMBINED RESIDUAL PLOTS (LEFT: SCATTER, RIGHT: HISTOGRAM)
    print("\nGenerating Combined Residual Analysis Plot...")
    
    # Residuals hesaplama (Eğer yukarıda hesaplanmadıysa burada hesaplar)
    residuals = all_y_true - all_y_pred

    # 1 Satır, 2 Sütunluk bir şekil oluşturuyoruz
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- SOL GRAFİK: Residuals vs Predicted (Scatter) ---
    axes[0].scatter(all_y_pred, residuals, s=45, edgecolor="white", alpha=0.7, color='steelblue')
    axes[0].axhline(0, color="#333333", linestyle="--", linewidth=2) # Sıfır çizgisi
    
    axes[0].set_xlabel("Predicted Quality of Sleep Score", fontsize=12)
    axes[0].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    axes[0].set_title("Residuals vs. Predicted Values", fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # --- SAĞ GRAFİK: Distribution of Residuals (Histogram) ---
    axes[1].hist(residuals, bins=20, edgecolor="black", alpha=0.8, color='mediumpurple')
    
    axes[1].set_xlabel("Residuals (Error)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Distribution of Residuals", fontsize=14, fontweight='bold')
    axes[1].grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"dt_residuals_combined_{current_time}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Residuals Histogram ---
   # 5) MANUAL 10-FOLD LOOP FOR PLOTTING (V2.0 - DAHA KİBAR GÖRÜNÜM)
    print("\nRunning Manual 10-Fold for Detailed Visualization...")
    
    # Grafik Boyutu ve Stili
    plt.figure(figsize=(10, 8)) # Boyutu biraz daha kompakt yaptık
    sns.set_style("whitegrid")  # Arka plan çizgileri kalsın
    
    # Marker ve Renkler
    markers = ['o', 's', '^', 'v', '<', '>', 'P', 'X', 'D', '*'] 
    colors = sns.color_palette("tab10", 10)
    
    all_y_true = []
    all_y_pred = []
    
    for fold_idx, (train_index, test_index) in enumerate(cv_strategy.split(features, target)):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]
        
        best_model.fit(X_train, y_train)
        fold_preds = best_model.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(fold_preds)
        
        # --- ESTETİK AYARLAR (Burayı değiştirdik) ---
        # Jitter'ı azalttık (0.08 -> 0.04) ki noktalar çok dağılmasın
        jitter_x = np.random.normal(0, 0.04, size=len(y_test))
        jitter_y = np.random.normal(0, 0.04, size=len(fold_preds))
        
        plt.scatter(
            y_test + jitter_x, 
            fold_preds + jitter_y,
            label=f'Fold {fold_idx + 1}',
            color=colors[fold_idx],
            marker=markers[fold_idx],
            s=45,              # DEĞİŞTİ: Nokta boyutu 70'ten 45'e indi (daha kibar)
            alpha=0.6,         # DEĞİŞTİ: Şeffaflık arttı (daha modern durur)
            edgecolor='white', # DEĞİŞTİ: Siyah kenar yerine beyaz kenar (daha yumuşak)
            linewidth=0.5
        )

    # --- Metrik Hesaplamaları (Aynı kalıyor) ---
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    mae = mean_absolute_error(all_y_true, all_y_pred)
    rmse = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
    r2 = r2_score(all_y_true, all_y_pred)

    # --- GRAFİK SONLANDIRMA ---
    
    # İdeal Doğru
    global_min = min(min(all_y_true), min(all_y_pred))
    global_max = max(max(all_y_true), max(all_y_pred))
    plt.plot([global_min, global_max], [global_min, global_max], 
             color='#333333', linestyle='--', linewidth=2, label='Ideal Line (y=x)') # Koyu gri çizgi

    # Başlık ve Eksenler
    plt.title(f"Decision Tree Prediction Performance (10-Fold CV)\n$R^2$ = {r2:.3f} | RMSE = {rmse:.3f}", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Actual Quality of Sleep", fontsize=12)
    plt.ylabel("Predicted Quality of Sleep", fontsize=12)
    
    # Lejant Ayarı (Grafiğin içine sığdırmaya çalışalım veya daha sade yapalım)
    # 'ncol=2' yaparak lejantı iki sütun yapıyoruz, boyu kısalıyor.
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Iterations", ncol=1, frameon=True)
    
    # Ticks
    plt.xticks(np.arange(int(global_min), int(global_max)+1, 1))
    plt.yticks(np.arange(int(global_min), int(global_max)+1, 1))
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / f"dt_scatter_clean_english_{current_time}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 9) SAVE MODEL
    # Fit on full data one last time before saving
    best_model.fit(features, target)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()