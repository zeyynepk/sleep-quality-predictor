import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# AYARLAR
# ============================================================
sns.set(style="whitegrid")
save_dir = Path("modeller")
save_dir.mkdir(parents=True, exist_ok=True)

dataset_file_name = "Sleep_health_and_lifestyle_dataset.csv"
target_column = "Quality of Sleep"

# ============================================================
# VERİYİ GÜVENLİ OKU
# ============================================================
try:
    df = pd.read_csv(dataset_file_name)
except UnicodeDecodeError:
    df = pd.read_csv(dataset_file_name, encoding="utf-8", engine="python")

# Sütun isimleri temizlensin
df.columns = df.columns.str.strip()

print("\n=== Veri Şekli ===")
print(df.shape)

print("\n=== Eksik Değerler ===")
print(df.isnull().sum().sort_values(ascending=False))

print("\n=== Sayısal Sütun Özeti ===")
print(df.describe().T)

# ============================================================
# BLOOD PRESSURE AYIRMA (REGRESYON İÇİN GEREKLİ)
# ============================================================
if "Blood Pressure" in df.columns:
    bp_split = df["Blood Pressure"].astype(str).str.split("/", n=1, expand=True)
    df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
    df.drop(columns=["Blood Pressure"], inplace=True)

# ============================================================
# HEDEF DEĞİŞKEN DAĞILIMI
# ============================================================
if target_column in df.columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[target_column].dropna(), bins=10, kde=True)
    plt.title("Quality of Sleep Dağılımı")
    plt.xlabel("Sleep Quality Score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_dir / "eda_quality_of_sleep_hist.png", dpi=400)
    plt.close()

    print("\n=== Hedef Değişken İstatistikleri ===")
    print(df[target_column].describe())
else:
    print(f"Hedef sütun '{target_column}' bulunamadı!")

# ============================================================
# KATEGORİK ÖZET
# ============================================================
categorical_columns = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

print("\n=== Kategorik Sütunlar ===")
for col in categorical_columns:
    print(f"\n- {col} (farklı değer sayısı: {df[col].nunique()})")
    print(df[col].astype(str).value_counts().head(10))

# ============================================================
# KORELASYON MATRİSİ — Sayısal değerler ile (annot=True)
# ============================================================
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(numeric_columns) > 1:
    plt.figure(figsize=(10, 8))
    corr = df[numeric_columns].corr()
    sns.heatmap(
        corr,
        cmap="coolwarm",
        annot=True,           # SAYISAL DEĞERLER EKRANA YAZILIR
        fmt=".2f",            # 2 ondalık
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_dir / "eda_numeric_correlation_heatmap_annot.png", dpi=400)
    plt.close()

# ============================================================
# AYKIRI DEĞER ANALİZİ (IQR) — Yuvarlama YOK
# ============================================================
def iqr_rate(series: pd.Series):
    series = series.dropna()
    if series.empty:
        return np.nan
    q1, q3 = np.percentile(series, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).mean()

# Ham yüzde değerler (round YOK)
outlier_summary = {col: iqr_rate(df[col]) * 100 for col in numeric_columns}

print("\n=== Aykırı Değer Oranları (%) — Ham Değerler ===")
for col, rate in outlier_summary.items():
    print(f"{col}: {rate:.6f}")

# Grafik
outlier_series = pd.Series(outlier_summary).sort_values(ascending=False)
plt.figure(figsize=(8, 5))
outlier_series.plot(kind="barh")
plt.title("Outlier Rates (IQR Method, %)")
plt.xlabel("Outlier Percentage")
plt.tight_layout()
plt.savefig(save_dir / "eda_outlier_rates_bar.png", dpi=400)
plt.close()

print("\nEDA tamamlandı. Tüm grafikler 'modeller' klasörüne kaydedildi.")
