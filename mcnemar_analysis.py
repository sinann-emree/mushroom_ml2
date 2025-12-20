import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar
import tensorflow as tf
import os

# 1. Veriyi Yükle
processed_data_path = 'data/mushrooms_processed.csv'
if not os.path.exists(processed_data_path):
    print("Hata: 'data/mushrooms_processed.csv' bulunamadı. Lütfen önce preprocessing.py çalıştır.")
    exit()

df = pd.read_csv(processed_data_path)
X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelleri tanımla
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier()
}

preds = {}

print("--- Modeller Eğitiliyor (McNemar Analizi için) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)
    print(f"{name} eğitildi.")

# 2. YSA Modelini hızlıca burada eğitelim (Dosyadan yükleme hatasını önlemek için)
ysa_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
ysa_model.compile(optimizer='adam', loss='binary_crossentropy')
ysa_model.fit(X_train, y_train, epochs=20, verbose=0) 

preds["ANN_YSA"] = (ysa_model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
print("ANN_YSA eğitildi.")

def calculate_mcnemar(y_true, pred1, pred2):
    # b: 1. model doğru, 2. model yanlış
    # c: 1. model yanlış, 2. model doğru
    b = np.sum((pred1 == y_true) & (pred2 != y_true))
    c = np.sum((pred1 != y_true) & (pred2 == y_true))
    
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result.pvalue

# Kıyaslanacak Çiftler
comparisons = [
    ("LogisticRegression", "RandomForest"),
    ("DecisionTree", "ANN_YSA"),
    ("RandomForest", "ANN_YSA")
]

print("\n--- McNemar Testi Sonuçları ---")
for m1, m2 in comparisons:
    p_val = calculate_mcnemar(y_test.values, preds[m1], preds[m2])
    status = "Anlamlı Fark Var" if p_val < 0.05 else "Anlamlı Fark Yok"
    print(f"{m1} vs {m2} -> p-value: {p_val:.5f} ({status})")