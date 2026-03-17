import streamlit as ui            # Arayüz oluşturmak için Streamlit
import pandas as data             # Model inputu için DataFrame üretmek amacıyla
import numpy as number            # Sayısal işlemler ve clipping için
import joblib                     # Sklearn modellerini yüklemek için
from pathlib import Path          # Platform bağımsız dosya yolu yönetimi
from typing import List, Dict     # Tip güvenliği için
import tensorflow as tf           # Derin öğrenme modellerini yüklemek için


models_dir = Path(".")

# Sayfa konfigürasyonu:
# layout="wide" : geniş ekran düzeni (iki sütun için ideal)
ui.set_page_config(page_title="Sleep Quality Predictor", layout="wide")



# CACHE'Lİ YÜKLEYİCİLER

# TensorFlow sürüm farklarından kaynaklanan LSTM layer config uyumsuzluğunu önlemek için özel bir LSTM sınıfı tanımlıyoruz.
class PatchedLSTM(tf.keras.layers.LSTM):
    @classmethod
    def from_config(cls, cfg):
        # Eski sürümlerde bulunan fakat yeni sürümde desteklenmeyen parametreyi temizliyoruz.
        cfg.pop("time_major", None)
        return super().from_config(cfg)


# Custom object sözlüğü (model load sırasında kullanılacak)
custom_objects = {
    "LSTM": PatchedLSTM,
}




# MODEL YÜKLEME FONKSİYONLARI (CACHE’Lİ)

# @ui.cache_resource: Model yalnızca ilk çağrıldığında RAM'e yüklenir. Sonraki çağrılarda tekrar yüklenmez → performans artar.

@ui.cache_resource
def load_cached_model(model_path_str: str):
    return tf.keras.models.load_model(
        model_path_str,
        compile=False,              # Eğitim compile bilgisine gerek yok
        custom_objects=custom_objects
    )

@ui.cache_resource
def load_cached_preprocessor(preproc_path_str: str):
    return joblib.load(preproc_path_str)

@ui.cache_resource
def load_cached_pipeline(pipeline_path_str: str):
    return joblib.load(pipeline_path_str)


# -------------------------------------------------
# YARDIMCI FONKSİYONLAR (POST-PROCESSING)
# -------------------------------------------------

def clamp_0_9(x: float) -> float:
    
    #Bu fonksiyon çıktıyı 0 ile 9 arasına sınırlar.
    return float(number.clip(x, 0.0, 9.0))

def pct_from_score(score: float) -> float:

    """
    0-9 ölçeğini yüzdeye çevirir.
    Örneğin 9 → 100%
    """
    return float(number.clip((score / 9.0) * 100.0, 0.0, 100.0))

def explain_percent(p: int) -> str:

    """
    Yüzdelik değere göre kullanıcıya yorum döndürür.
    Bu katman UX (User Experience) katmanıdır.
    """
    if p < 60:
        return "Low sleep quality."
    elif p < 75:
        return "Moderate sleep quality."
    elif p < 85:
        return "Good sleep quality."
    else:
        return "Very good sleep quality."



# INPUT → MODEL FORMATINA DÖNÜŞTÜRME

def compose_row(
    gender: str, age: int, occupation: str, sleep_dur: float,
    stress: int, pal: int, bmi_cat: str, hr: int, steps: int,
    bp_sys: int, bp_dia: int
) -> data.DataFrame:
    """
    Kullanıcıdan alınan tüm inputları,
    eğitim sırasında kullanılan kolon isimleriyle
    eşleşen tek satırlık bir pandas DataFrame'e dönüştürür.
    """
    row = {
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration": sleep_dur,
        "Stress Level": stress,
        "Physical Activity Level": pal,
        "BMI Category": bmi_cat,
        "Heart Rate": hr,
        "Daily Steps": steps,
        "BP_Systolic": bp_sys,
        "BP_Diastolic": bp_dia,
    }

    # Model predict fonksiyonu 2D yapı beklediği için
    # tek satırlık DataFrame döndürüyoruz.
    return data.DataFrame([row])


# -------------------------------------------------
# MODEL DOSYALARI (INTERNAL ANAHTARLAR)
# -------------------------------------------------

sk_models: Dict[str, Path] = {
    "Rastgele Orman":     models_dir / "rf_regressor_pipeline.joblib",
    "KNN":                models_dir / "knn_regressor_pipeline.joblib",
    "SVR (RBF)":          models_dir / "svr_regressor_pipeline.joblib",
    "Karar Ağacı":        models_dir / "dt_regressor_pipeline.joblib",
    "Doğrusal Regresyon": models_dir / "lr_regressor_pipeline.joblib",
}

ann_files = {
    "model": models_dir / "ann_regressor_model.h5",
    "pre":   models_dir / "ann_regressor_preprocessor.joblib",
}

cnn_files = {
    "model": models_dir / "cnn_regressor_model.h5",
    "pre":   models_dir / "cnn_regressor_preprocessor.joblib",
}

lstm_files = {
    "model": models_dir / "lstm_regressor_model.h5",
    "pre":   models_dir / "lstm_regressor_preprocessor.joblib",
}



# SIDEBAR (KULLANICIYA GÖRÜNEN MODEL İSİMLERİ)

ui.sidebar.header("Model Selection")

model_options = [
    "Random Forest",
    "KNN",
    "SVR",
    "Decision Tree",
    "Linear Regression",
    "ANN",
    "CNN",
    "LSTM",
]

model_choice_display = ui.sidebar.radio("Select model", model_options)

# UI'daki isimleri, internal anahtarlara eşliyoruz.
model_name_map = {
    "Random Forest": "Rastgele Orman",
    "KNN": "KNN",
    "SVR": "SVR (RBF)",
    "Decision Tree": "Karar Ağacı",
    "Linear Regression": "Doğrusal Regresyon",
    "ANN": "ANN (Yoğun)",
    "CNN": "CNN (Conv1D)",
    "LSTM": "LSTM (Sıra)",
}

model_choice = model_name_map[model_choice_display]



# ANA ARAYÜZ

ui.title("Sleep Quality Prediction App")

# İki sütunlu tasarım 
c1, c2 = ui.columns(2)

with c1:
    gender_tr = ui.selectbox("Gender", ["Female", "Male", "Other"])
    age = ui.slider("Age", 18, 90, 42)
    occupation = ui.selectbox(
        "Occupation",
        ["Accountant", "Doctor", "Teacher", "Engineer",
         "Sales Representative", "Writer", "Lawyer",
         "Nurse", "Manager", "Software Engineer"]
    )
    bmi_cat = ui.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])
    heart = ui.slider("Heart Rate (bpm)", 40, 140, 70)
    sleep_dur = ui.number_input("Sleep Duration (hours)", 3.0, 12.0, 7.2, step=0.1)

with c2:
    stress = ui.slider("Stress Level (1-10)", 1, 10, 5)
    pal = ui.slider("Physical Activity Level (0-100)", 0, 100, 60)
    steps = ui.number_input("Daily Steps", 0, 50000, 7000, step=100)
    bp_sys = ui.slider("Systolic BP", 90, 200, 125)
    bp_dia = ui.slider("Diastolic BP", 50, 130, 80)

ui.markdown("---")



# KERAS MODEL TAHMİN FONKSİYONU

def keras_predict(model_path, preproc_path, df_raw, reshape=None):
    """
    Derin öğrenme modelleri için:
    1. Preprocessor yüklenir
    2. Input transform edilir
    3. Gerekirse reshape yapılır
    4. Model tahmini alınır
    """

    pre = load_cached_preprocessor(str(preproc_path))
    Xs = pre.transform(df_raw)

    # TensorFlow için float32 formatına dönüştürüyoruz
    Xs = number.asarray(Xs, dtype=number.float32)

    # CNN ve LSTM için shape uyarlaması

    # CNN ve LSTM modelleri 3 boyutlu tensor bekler.
    # Preprocessor’dan çıkan veri 2 boyutludur: (1, feature_sayısı).
    # expand_dims ile modele uygun olacak şekilde ekstra boyut ekliyoruz.
    # -1 → son eksene boyut ekler  → (1, F) → (1, F, 1)
    #  1 → ortadaki eksene boyut ekler → (1, F) → (1, 1, F)
    if reshape == "F_1":
        Xs = number.expand_dims(Xs, -1)
    elif reshape == "T_F":
        Xs = number.expand_dims(Xs, 1)

    model = load_cached_model(str(model_path))
    y = model.predict(Xs, verbose=0)

    return float(number.ravel(y)[0])



# TAHMİN BUTONU

if ui.button("🧠 Tahmin et", type="primary"):

    # Kullanıcı girdilerini model formatına dönüştürüyoruz
    df_raw = compose_row(
        gender_tr, int(age), occupation, float(sleep_dur),
        int(stress), int(pal), bmi_cat, int(heart), int(steps),
        int(bp_sys), int(bp_dia)
    )

    try:

        # Eğer sklearn pipeline modeli seçildiyse
        if model_choice in sk_models:
            pipeline = load_cached_pipeline(str(sk_models[model_choice]))
            raw_value = float(pipeline.predict(df_raw)[0])

        # ANN modeli
        elif model_choice == "ANN (Yoğun)":
            raw_value = keras_predict(
                ann_files["model"],
                ann_files["pre"],
                df_raw
            )

        # CNN modeli
        elif model_choice == "CNN (Conv1D)":
            raw_value = keras_predict(
                cnn_files["model"],
                cnn_files["pre"],
                df_raw,
                reshape="F_1"
            )

        # LSTM modeli
        else:
            try:
                raw_value = keras_predict(
                    lstm_files["model"],
                    lstm_files["pre"],
                    df_raw,
                    reshape="F_1"
                )
            except Exception:
                raw_value = keras_predict(
                    lstm_files["model"],
                    lstm_files["pre"],
                    df_raw,
                    reshape="T_F"
                )

      
        y_pred = round(clamp_0_9(raw_value))
        pct = round(pct_from_score(y_pred))

        ui.success(f"Predicted Sleep Quality: {y_pred} / 9  →  {pct}%")
        ui.caption(f"Raw model output: {raw_value:.3f}")
        ui.info(explain_percent(pct))

    except Exception as e:
        # Olası model yükleme veya tahmin hatalarını gösterir
        ui.exception(e)