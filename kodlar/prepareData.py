import pandas as data
from typing import List, Tuple, Optional
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------------------------------------
# PROJE KAPSAMI:
# Bu modul, "Quality of Sleep" Hedef Degiskenini
# 4–9 araliginda SUREKLI bir skor olarak alan
# REGRESYON problemleri icin veri hazirlar.
# ----------------------------------------------------

# Varsayılan dosya adı ve hedef sütun
csv_path = "Sleep_health_and_lifestyle_dataset.csv"
target = "Quality of Sleep"


def _resolve_csv_path(csv_path_str: str) -> Path:
    """
    csv_path parametresini güvenli bir şekilde çözer:
    - Önce mevcut çalışma dizininde arar,
    - Bulamazsa prepareData.py dosyasının bir üst klasöründe (proje kökü) arar.
    """
    p = Path(csv_path_str)
    if p.is_file():
        return p

    # prepareData.py konumuna göre proje kökünü bul
    root = Path(__file__).resolve().parents[1]
    alt = root / csv_path_str
    if alt.is_file():
        return alt

    # Son çare: olduğu gibi döndür (hata alırsan mesajdan anlarsın)
    return p


# ----------------------------------------------------
# Veriyi okur ve temizler (REGRESYON için)
# ----------------------------------------------------
def readAndClean(
    csv_path: str = csv_path,
    target: str = target,
    columns_to_drop: Optional[List[str]] = None
) -> Tuple[data.DataFrame, data.Series]:
    """
    CSV dosyasını okur, temel temizlikleri yapar, hedef sütunu ayırır.

    - Blood Pressure → BP_Systolic / BP_Diastolic (iki sayısal sütun)
    - Person ID, Sleep Disorder gibi sızıntı yaratabilecek sütunları atar.
    - Hedef sütun (Quality of Sleep) SUREKLI bir sayısal degiskendir
      ve REGRESYON icin sayisal tipe cevrilir.
    """
    csv_file = _resolve_csv_path(csv_path)

    # Güvenli okuma (encoding sorunu olursa fallback)
    try:
        data_frame = data.read_csv(csv_file)
    except UnicodeDecodeError:
        data_frame = data.read_csv(csv_file, encoding="utf-8", engine="python")

    # Sütun adlarındaki boşlukları temizler
    data_frame.columns = data_frame.columns.str.strip()

    # Blood Pressure varsa ikiye böler (or: "126/83" → 126 ve 83)
    if "Blood Pressure" in data_frame.columns:
        blood_pressure_split = (
            data_frame["Blood Pressure"]
            .astype(str)
            .str.split("/", n=1, expand=True)
        )
        data_frame["BP_Systolic"] = data.to_numeric(
            blood_pressure_split[0], errors="coerce"
        )
        data_frame["BP_Diastolic"] = data.to_numeric(
            blood_pressure_split[1], errors="coerce"
        )
        data_frame = data_frame.drop(columns=["Blood Pressure"])

    # Gereksiz veya sızıntı yaratabilecek sütunlar
    # Not: Sleep Disorder ileride AYRI bir sınıflandırma problemi için kullanılabilir,
    # bu projede regresyon hedefi Quality of Sleep oldugu icin atiliyor.
    default_drop_columns = ["Person ID", "Sleep Disorder"]
    if columns_to_drop is not None:
        for c in columns_to_drop:
            if c not in default_drop_columns:
                default_drop_columns.append(c)

    for column_name in default_drop_columns:
        if column_name in data_frame.columns:
            data_frame = data_frame.drop(columns=column_name)

    # Hedef sütunun gerçekten var olduğundan emin ol
    if target not in data_frame.columns:
        raise KeyError(
            f"Hedef sütun bulunamadı: '{target}'. "
            f"Mevcut sütunlar: {list(data_frame.columns)}"
        )

    # Özellikleri (X) ve hedefi (y) ayırır
    features = data_frame.drop(columns=[target])
    target_values = data_frame[target]

    # ------------------------------------------------
    # KRITIK NOKTA:
    # Hedefi REGRESYON icin sayisal tipe ceviriyoruz.
    # Böylece model her zaman surekli bir QoS skoru tahmin eder.
    # ------------------------------------------------
    target_values = data.to_numeric(target_values, errors="coerce")

    # Hedefi NaN olan satırları güvenli şekilde at
    valid_mask = target_values.notna()
    features = features.loc[valid_mask].reset_index(drop=True)
    target_values = target_values.loc[valid_mask].reset_index(drop=True)

    return features, target_values


# ----------------------------------------------------
# Ön-işleme pipeline'ı oluşturur
# ----------------------------------------------------
def createPreprocessor(features: data.DataFrame) -> ColumnTransformer:
    """
    - Sayısal sütunlar: median ile doldur + StandardScaler
    - Kategorik sütunlar: en sık değer ile doldur + OneHotEncoder (unknown=ignore)

    Bu preprocessor, tum REGRESYON modellerinde aynen kullanilabilir:
        Pipeline([("pre", preprocessor), ("reg", model)])
    """
    # Sayısal ve kategorik sütunları ayırır
    numeric_feature_names = features.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_feature_names = features.select_dtypes(
        exclude=["int64", "float64"]
    ).columns.tolist()

    # Sayısal veriler için: median doldurma + standardizasyon
    numeric_preprocess_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    # Kategorik veriler için: en sık görülenle doldur + OneHotEncoder
    categorical_preprocess_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # İki pipeline'ı bir ColumnTransformer içinde birleştirir
    preprocessor_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess_pipeline, numeric_feature_names),
            ("cat", categorical_preprocess_pipeline, categorical_feature_names),
        ]
    )
    return preprocessor_transformer


# ----------------------------------------------------
# Tüm süreci tek adımda çalıştırır
# ----------------------------------------------------
def prepareData(csv_path: str = csv_path, target: str = target):
    """
    Kullanım:
        X, y, pre = prepareData()

    DÖNDÜRDÜKLERİ:
        - X  : Özellikler (sayısal + one-hot kategorik)
        - y  : Quality of Sleep (4–9 arası SUREKLI skor, REGRESYON hedefi)
        - pre: ColumnTransformer (ön-isleme pipeline'i)

    Sonrasında:
        Pipeline([("pre", pre), ("reg", model)]) ile
        train / cross-validation / oof vb. REGRESYON islemleri yapilabilir.
    """
    features, target_values = readAndClean(csv_path, target)
    preprocessor_transformer = createPreprocessor(features)
    return features, target_values, preprocessor_transformer


# ----------------------------------------------------
# Dosya direkt çalıştırıldığında küçük bir özet verir
# ----------------------------------------------------
if __name__ == "__main__":
    X, y, pre = prepareData()
    print("X shape:", X.shape)
    print("y length:", len(y))
    print("Sample columns:", list(X.columns)[:8])
    print("Target min/max:", float(y.min()), float(y.max()))
