import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import sys

# --- PATH VE SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prepareData import prepareData

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

BASE_DIR = Path(os.path.dirname(__file__)).parent
SAVE_DIR = BASE_DIR / "modeller"
MODEL_PATH = SAVE_DIR / "dt_regressor_final.joblib"
IMG_SAVE_DIR = SAVE_DIR 

def main():
    print("1. Veri ve Model Yükleniyor...")
    X, y, _ = prepareData()
    
    full_pipeline = joblib.load(MODEL_PATH)
    preprocessor = full_pipeline.named_steps["pre"]
    model = full_pipeline.named_steps["reg"]

    print("2. Veri Ön İşleme Yapılıyor...")
    X_transformed = preprocessor.transform(X)
    raw_feature_names = preprocessor.get_feature_names_out()
    
    # --- İSİM DÜZELTME (Hocanın İsteği) ---
    # "num__Sleep_Duration" -> "Sleep Duration"
    clean_feature_names = [
        name.replace("num__", "").replace("cat__", "").replace("_", " ") 
        for name in raw_feature_names
    ]
    
    # DataFrame'i temiz isimlerle oluşturuyoruz
    X_transformed_df = pd.DataFrame(X_transformed, columns=clean_feature_names)

    print("3. SHAP Değerleri Hesaplanıyor...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed_df)

    # =========================================================================
    # GRAFİK 1: FEATURE IMPORTANCE (BAR PLOT) - SAYISAL DEĞERLİ
    # =========================================================================
    print("4. Grafik 1: Özellik Önem Düzeyi (Bar Plot) oluşturuluyor...")
    
    plt.figure(figsize=(10, 6)) # Boyutu biraz daha kompakt yaptık
    
    # max_display=8 (Hocanın isteği: Sadece 8 özellik)
    shap.summary_plot(
        shap_values, 
        X_transformed_df, 
        plot_type="bar", 
        show=False,
        max_display=8,
        color="#008bfb" # Standart SHAP mavisi
    )
    
    # --- DEĞERLERİ YAZDIRMA KISMI ---
    ax = plt.gca()
    
    # Yazıların sığması için sağ tarafı %15 genişlet
    x_min, x_max = ax.get_xlim()
    ax.set_xlim(x_min, x_max * 1.2)

    for p in ax.patches:
        width = p.get_width()
        y_pos = p.get_y() + p.get_height() / 2
        
        # Sadece 0'dan büyükse yazdır
        if width > 0:
            ax.text(
                width + (x_max * 0.01), 
                y_pos, 
                f'{width:.3f}', 
                va='center', 
                fontsize=11, 
                color='black',
                weight='bold'
            )
    
    plt.xlabel("Ortalama SHAP Değeri (Etki Büyüklüğü)")
    # plt.title(...) SATIRINI SİLDİK (Hocanın isteği)
    
    plt.tight_layout()
    plt.savefig(IMG_SAVE_DIR / "dt_shap_importance_clean.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Kaydedildi: {IMG_SAVE_DIR / 'dt_shap_importance_clean.png'}")

    # =========================================================================
    # GRAFİK 2: ÖZET (BEESWARM PLOT) - Opsiyonel (Yedek bulunsun)
    # =========================================================================
    print("5. Grafik 2: Etki Yönü Özeti (Beeswarm Plot) oluşturuluyor...")
    plt.figure(figsize=(10, 6))
    
    shap.summary_plot(
        shap_values, 
        X_transformed_df, 
        show=False,
        max_display=8 # Burayı da 8 yaptık
    )
    
    # plt.title(...) SİLDİK
    plt.tight_layout()
    plt.savefig(IMG_SAVE_DIR / "dt_shap_beeswarm_clean.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Kaydedildi: {IMG_SAVE_DIR / 'dt_shap_beeswarm_clean.png'}")
    
    print("\nİşlem Başarıyla Tamamlandı.")

if __name__ == "__main__":
    main()