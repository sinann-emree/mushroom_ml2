import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Klasör kontrolü
if not os.path.exists('models'): os.makedirs('models')

# 1. Veriyi yükle
df = pd.read_csv('data/mushrooms.csv')
if 'veil-type' in df.columns: df = df.drop(['veil-type'], axis=1)

# 2. Encoderları eğit (Sadece veride olan harfleri öğrenir)
encoders = {}
X_encoded = pd.DataFrame()

for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('class', axis=1)
y = df['class']

# 3. Modeli eğit ve KAYDET
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/best_model.pkl')
joblib.dump(encoders, 'models/encoders.pkl')

print("BAŞARILI: Model ve veriler hazırlandı. Şimdi app.py çalıştırabilirsin.")