from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap


# ------------------------------------------------------------
# 1) Yollar
# ------------------------------------------------------------
path_root = Path(__file__).resolve().parents[1]
models_dir = path_root / "modeller"

rf_path = models_dir / "rf_regressor_pipeline.joblib"
print("RF pipeline:", rf_path)

pipeline = joblib.load(rf_path)
pre = pipeline.named_steps["pre"]
reg = pipeline.named_steps["reg"]

# ------------------------------------------------------------
# 2) Veri dosyasını bul
#    (Bu satır hata verirse, dosyanın gerçek yolunu yazacağız)
# ------------------------------------------------------------
veri_yol = path_root / "Sleep_health_and_lifestyle_dataset.csv"
print("Veri yolu:", veri_yol)

df = pd.read_csv(veri_yol)
# ------------------------------------------------------------
# Pipeline bu iki sütunu bekliyor: BP_Systolic, BP_Diastolic
# CSV'de "Blood Pressure" varsa onu "120/80" gibi formattan ayırıyoruz.
# ------------------------------------------------------------
if "Blood Pressure" in df.columns:
    bp_split = df["Blood Pressure"].astype(str).str.split("/", expand=True)

    # Güvenli dönüşüm (hatalı varsa NaN olur)
    df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")

    # Orijinal sütunu kaldır (pipeline bunu istemiyor olabilir)
    df = df.drop(columns=["Blood Pressure"])

print("Blood Pressure split OK. Mevcut sütunlar:", list(df.columns))


hedef = "Quality of Sleep"
X = df.drop(columns=[hedef]).copy()
y = df[hedef].copy()

print("Veri shape:", df.shape)

# ------------------------------------------------------------
# 3) SHAP için örneklem (hız için)
# ------------------------------------------------------------
ornek_sayi = min(400, len(X))
X_ornek = X.sample(ornek_sayi, random_state=42)

# Ön işleme sonrası matris
X_tr = pre.transform(X_ornek)

# Feature isimleri (OneHot sonrası)
try:
    feature_names = pre.get_feature_names_out()
except Exception:
    feature_names = [f"feature_{i}" for i in range(X_tr.shape[1])]

print("Transform shape:", X_tr.shape)

# ------------------------------------------------------------
# 4) SHAP hesapla
# ------------------------------------------------------------
explainer = shap.TreeExplainer(reg)
shap_values = explainer.shap_values(X_tr)

# ------------------------------------------------------------
# 5) Grafikler (modeller/ klasörüne kaydet)
# ------------------------------------------------------------
out_bar = models_dir / "shap_rf_importance_bar.png"
out_bee = models_dir / "shap_rf_summary_beeswarm.png"

plt.figure()
shap.summary_plot(shap_values, X_tr, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(out_bar, dpi=400)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_tr, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(out_bee, dpi=400)
plt.close()

print("\nKayıt edildi:")
print(" -", out_bar)
print(" -", out_bee)

# ------------------------------------------------------------
# 6) İlk 10 özelliği terminale yaz
# ------------------------------------------------------------
mean_abs = np.abs(shap_values).mean(axis=0)
idx = np.argsort(mean_abs)[::-1][:10]

print("\nİlk 10 özellik (SHAP mean|value|):")
for r, i in enumerate(idx, 1):
    print(f"{r:02d}) {feature_names[i]} = {mean_abs[i]:.6f}")
