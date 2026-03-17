from pathlib import Path
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from sklearn.pipeline import Pipeline


# ============================================================
# AYARLAR (gerekirse burada oynarsın)
# ============================================================
HEDEF = "Quality of Sleep"
VERI_DOSYA_ADI = "Sleep_health_and_lifestyle_dataset.csv"

# KernelExplainer yavaş olduğundan örneklem küçük tutuyoruz:
N_BACKGROUND = 50   # explainer için arka plan
N_EXPLAIN = 80      # SHAP hesaplanacak örnek sayısı

# SHAP örnekleme hız parametresi (daha düşük = daha hızlı, daha az hassas)
NSAMPLES = 200

# ============================================================
# YOLLAR
# ============================================================
path_root = Path(__file__).resolve().parents[1]
models_dir = path_root / "modeller"
lr_path = models_dir / "linear_regression_pipeline.joblib"

print("Proje kökü:", path_root)
print("LR pipeline:", lr_path)

# ============================================================
# PIPELINE YÜKLE
# ============================================================
pipeline = joblib.load(lr_path)

if not isinstance(pipeline, Pipeline):
    raise TypeError("linear_regression_pipeline.joblib bir sklearn Pipeline değil gibi görünüyor.")

print("\nPipeline adımları:")
for name, step in pipeline.named_steps.items():
    print(f" - {name}: {type(step)}")

# Bizde isimler genelde: pre, reg
pre = pipeline.named_steps.get("pre", None)
reg = pipeline.named_steps.get("reg", None)

if pre is None or reg is None:
    raise KeyError("Pipeline içinde 'pre' ve 'reg' adımları bulunamadı. Adım isimlerini yukarıdaki çıktıda kontrol et.")

# ============================================================
# VERİYİ OKU
# ============================================================
veri_yol = path_root / VERI_DOSYA_ADI
print("\nVeri yolu:", veri_yol)

df = pd.read_csv(veri_yol)

# ------------------------------------------------------------
# Blood Pressure split: pipeline BP_Systolic & BP_Diastolic bekliyor
# ------------------------------------------------------------
if "Blood Pressure" in df.columns:
    bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
    df["BP_Systolic"] = pd.to_numeric(bp[0], errors="coerce")
    df["BP_Diastolic"] = pd.to_numeric(bp[1], errors="coerce")
    df = df.drop(columns=["Blood Pressure"])

if HEDEF not in df.columns:
    raise ValueError(f"Hedef sütun bulunamadı: {HEDEF}. Sende hedef adı farklıysa HEDEF'i değiştir.")

X = df.drop(columns=[HEDEF]).copy()
y = df[HEDEF].copy()

print("Veri shape:", df.shape)
print("X shape:", X.shape, "| y shape:", y.shape)

# ============================================================
# ÖRNEKLE (KernelExplainer hız için)
# ============================================================
n_bg = min(N_BACKGROUND, len(X))
n_exp = min(N_EXPLAIN, len(X))

X_bg = X.sample(n_bg, random_state=42)
X_exp = X.sample(n_exp, random_state=7)

# Transform
X_bg_tr = pre.transform(X_bg)
X_exp_tr = pre.transform(X_exp)

# Feature isimleri (one-hot sonrası)
try:
    feature_names = pre.get_feature_names_out()
except Exception:
    feature_names = [f"feature_{i}" for i in range(X_bg_tr.shape[1])]

print("\nTransform shape (background):", X_bg_tr.shape)
print("Transform shape (explain):   ", X_exp_tr.shape)

# ============================================================
# SHAP (Linear Regression için KernelExplainer)
# ============================================================
# reg.predict, pre-transform edilmiş veriyi bekliyor.
# Biz reg'i direkt kullandığımız için X_tr vermek zorundayız.
print("\nSHAP KernelExplainer başlatılıyor (bu kısım RF'ye göre daha yavaş olabilir)...")

explainer = shap.KernelExplainer(reg.predict, X_bg_tr)
shap_values = explainer.shap_values(X_exp_tr, nsamples=NSAMPLES)

# ============================================================
# GRAFİKLERİ KAYDET
# ============================================================
out_bar = models_dir / "shap_lr_importance_bar.png"
out_bee = models_dir / "shap_lr_summary_beeswarm.png"

plt.figure()
shap.summary_plot(shap_values, X_exp_tr, feature_names=feature_names, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(out_bar, dpi=400, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_exp_tr, feature_names=feature_names, show=False)
plt.tight_layout()
plt.savefig(out_bee, dpi=400, bbox_inches="tight")
plt.close()

print("\nKayıt edildi:")
print(" -", out_bar)
print(" -", out_bee)

# ============================================================
# İLK 10 ÖZELLİĞİ YAZDIR
# ============================================================
mean_abs = np.abs(shap_values).mean(axis=0)
idx = np.argsort(mean_abs)[::-1][:10]

print("\nİlk 10 özellik (Linear Regression, SHAP mean|value|):")
for r, i in enumerate(idx, 1):
    print(f"{r:02d}) {feature_names[i]} = {mean_abs[i]:.6f}")

print("\nBitti.")
