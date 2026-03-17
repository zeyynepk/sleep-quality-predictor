import pandas as pd
import matplotlib.pyplot as plot
from pathlib import Path

# =====================================
# 1) KAYIT KLASORU
# =====================================
save_dir = Path("modeller")
save_dir.mkdir(parents=True, exist_ok=True)
dpi_value = 400

# =====================================
# 2) MODEL SONUCLARI TABLOSU
#    (10-fold OOF sonuclarindan gelen degerler)
# =====================================
regression_results = pd.DataFrame([
    ("Decision Tree",     0.967, 0.965, 0.069, 0.216, 0.967, 81.29),
    ("Random Forest",     0.983, 0.982, 0.047, 0.158, 0.983, 81.28),
    ("KNN",               0.978, 0.977, 0.032, 0.177, 0.978, 81.16),
    ("Linear Regression", 0.949, 0.947, 0.155, 0.270, 0.949, 81.24),
    ("SVR (RBF)",         0.977, 0.976, 0.038, 0.183, 0.977, 81.25),
    ("ANN (Dense)",       0.883, 0.875, 0.282, 0.409, 0.898, 79.53),
    ("CNN (Conv1D)",      0.927, 0.922, 0.197, 0.322, 0.931, 80.34),
    ("LSTM",              0.970, 0.967, 0.099, 0.209, 0.970, 81.22),
], columns=["Model", "R2", "AdjR2", "MAE", "RMSE", "EVS", "MeanQoS"]).set_index("Model")

print("=== Regression Models — Summary (10-fold OOF) ===")
print(regression_results.round(3))


# =====================================
# 3) TEK FIGUR ICINDE 3 METRIK: R2, MAE, RMSE
# =====================================
fig, axes = plot.subplots(1, 3, figsize=(15, 5))

metric_info = [
    ("R2",   "R²", True),   # buyuk iyi
    ("MAE",  "MAE", False),  # kucuk iyi
    ("RMSE", "RMSE",False),  # kucuk iyi
]

for ax, (col, title, higher_is_better) in zip(axes, metric_info):
    series = regression_results[col]

    # Siralama
    if higher_is_better:
        series = series.sort_values(ascending=True)   # kucukten buyuge
    else:
        series = series.sort_values(ascending=False)  # buyukten kucuge

    series.plot(kind="barh", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(col)

    # Cubuklarin uzerine sayisal deger yaz
    for i, (model_name, value) in enumerate(series.items()):
        ax.text(
            value,
            i,
            f"{value:.3f}",
            va="center",
            ha="left",
            fontsize=8
        )

fig.suptitle("Regression Modelleri — R², MAE ve RMSE Karsilastirmasi", y=1.02)
plot.tight_layout()

summary_path = save_dir / "model_comparison_all_metrics.png"
fig.savefig(summary_path, dpi=dpi_value, bbox_inches="tight")
plot.show()
print(f"Tek figur icinde karsilastirma grafigi kaydedildi: {summary_path}")

# =====================================
# 4) MODELLERIN BASARISINI GOSTEREN EK GRAFIK
#    (En basarili model R²'ye gore vurgulanir)
# =====================================

# R2'ye gore sirala (yuksek daha iyi)
r2_series = regression_results["R2"].sort_values(ascending=True)

# En yuksek R2'li modeli bul
best_model = r2_series.idxmax()

# Renkleri ayarla: tum modeller gri, en iyisi baska renk
colors = ["gray"] * len(r2_series)
best_index = list(r2_series.index).index(best_model)
colors[best_index] = "tab:blue"

fig2, ax2 = plot.subplots(figsize=(8, 5))
bars = ax2.barh(r2_series.index, r2_series.values, color=colors)

ax2.set_title("Modellerin Basarisi — En Iyi Model (R²'ye Gore)")
ax2.set_xlabel("R²")

# Cubuklarin uzerine deger yaz
for bar in bars:
    width = bar.get_width()
    ax2.text(
        width,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}",
        va="center",
        ha="left",
        fontsize=8
    )

# En iyi modeli aciklayici not
ax2.text(
    0.50,
    0.05,
    f"En yuksek R² degerine sahip model: {best_model}",
    transform=ax2.transAxes,
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

plot.tight_layout()

best_path = save_dir / "model_best_by_R2.png"
fig2.savefig(best_path, dpi=dpi_value, bbox_inches="tight")
plot.show()
print(f"Modellerin basarisini gosteren grafik kaydedildi: {best_path}")
