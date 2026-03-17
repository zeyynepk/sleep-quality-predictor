import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# KLASİK MODELLER
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# DERİN ÖĞRENME İÇİN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LSTM, Input

# Rastgelelikleri sabitle (Sonuçlar her seferinde aynı çıksın)
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# 1. VERİ HAZIRLAMA (SİZİN STANDARTLARINIZ)
# =============================================================================
def get_data_ready(csv_path, target_col):
    try:
        df = pd.read_csv(csv_path)
    except:
        df = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    
    df.columns = df.columns.str.strip()
    
    # Blood Pressure Düzeltme
    if "Blood Pressure" in df.columns:
        bp_split = df["Blood Pressure"].astype(str).str.split("/", n=1, expand=True)
        df["BP_Systolic"] = pd.to_numeric(bp_split[0], errors="coerce")
        df["BP_Diastolic"] = pd.to_numeric(bp_split[1], errors="coerce")
        df.drop(columns=["Blood Pressure"], inplace=True)
        
    # Gereksizleri At
    drops = ["Person ID", "Sleep Disorder"]
    for d in drops:
        if d in df.columns:
            df.drop(columns=[d], inplace=True)
            
    # Hedef Kontrol
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def create_preprocessor(X):
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns
    
    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

# =============================================================================
# 2. VERİYİ YÜKLE VE İŞLE
# =============================================================================
dosya = "Sleep_health_and_lifestyle_dataset.csv"
hedef = "Quality of Sleep"  # Veya 'Sleep Duration'

print(f"Veri hazırlanıyor... Hedef: {hedef}")
X, y = get_data_ready(dosya, hedef)
preprocessor = create_preprocessor(X)

# Ön işleme (Scaling/Encoding) işlemini burada yapıyoruz ki DL modelleri için şekil (shape) alabilelim
X_processed = preprocessor.fit_transform(X)
y = y.values # Numpy array'e çevir

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f"Eğitim Verisi Boyutu: {X_train.shape}")
print("-" * 60)
print(f"{'Model':<20} | {'R2 Score':<10} | {'MAE':<10} | {'RMSE':<10}")
print("-" * 60)

# =============================================================================
# 3. KLASİK MODELLER (SCIKIT-LEARN)
# =============================================================================
classic_models = {
    "Linear Regression": LinearRegression(),
    "SVR (Default)": SVR(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "KNN (Default)": KNeighborsRegressor(),
    "ANN (MLP)": MLPRegressor(random_state=42, max_iter=2000)
}

for name, model in classic_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"{name:<20} | {r2:.4f}     | {mae:.4f}     | {rmse:.4f}")

# =============================================================================
# 4. DERİN ÖĞRENME MODELLERİ (CNN & LSTM - Default/Basit Yapı)
# =============================================================================
# Derin öğrenme için veriyi 3 boyutlu hale getirmeliyiz: (Örnek Sayısı, Zaman Adımı=1, Özellik Sayısı)
X_train_dl = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_dl = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
input_shape = (1, X_train.shape[1])

# --- CNN MODELİ (Basit) ---
cnn = Sequential([
    Input(shape=input_shape),
    Conv1D(filters=64, kernel_size=1, activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(1)
])
cnn.compile(optimizer='adam', loss='mse')
cnn.fit(X_train_dl, y_train, epochs=30, batch_size=16, verbose=0) # Sessiz mod
y_pred_cnn = cnn.predict(X_test_dl, verbose=0).flatten()

print(f"{'CNN (Default)':<20} | {r2_score(y_test, y_pred_cnn):.4f}     | {mean_absolute_error(y_test, y_pred_cnn):.4f}     | {np.sqrt(mean_squared_error(y_test, y_pred_cnn)):.4f}")

# --- LSTM MODELİ (Basit) ---
lstm = Sequential([
    Input(shape=input_shape),
    LSTM(50, activation='relu'),
    Dense(1)
])
lstm.compile(optimizer='adam', loss='mse')
lstm.fit(X_train_dl, y_train, epochs=30, batch_size=16, verbose=0)
y_pred_lstm = lstm.predict(X_test_dl, verbose=0).flatten()

print(f"{'LSTM (Default)':<20} | {r2_score(y_test, y_pred_lstm):.4f}     | {mean_absolute_error(y_test, y_pred_lstm):.4f}     | {np.sqrt(mean_squared_error(y_test, y_pred_lstm)):.4f}")
print("-" * 60)