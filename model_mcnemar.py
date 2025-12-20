import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.contingency_tables import mcnemar
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

def run_mcnemar():
    print("--- 2) MCNEMAR TESTİ ---")
    
    # 1. Verileri Yükle
    try:
        X_train = pd.read_csv('data/X_train_processed.csv')
        X_test = pd.read_csv('data/X_test_processed.csv')
        y_train = pd.read_csv('data/y_train_processed.csv').values.ravel()
        y_test = pd.read_csv('data/y_test_processed.csv').values.ravel()
    except FileNotFoundError:
        print("[HATA] Veriler bulunamadı.")
        return

    # 2. Modelleri Eğit (Kıyaslama için 3 farklı tip seçelim)
    # A) Basit Model: Lojistik Regresyon
    # B) Güçlü Model: Random Forest
    # C) Derin Model: YSA
    
    print("Modeller Eğitiliyor...")
    
    # Model A: Lojistik Regresyon
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train, y_train)
    pred_lr = model_lr.predict(X_test)
    
    # Model B: Random Forest
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    
    # Model C: YSA
    model_ann = Sequential([
        Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_ann.fit(X_train, y_train, epochs=50, verbose=0)
    pred_ann = (model_ann.predict(X_test).ravel() > 0.5).astype(int)

    # 3. McNemar Fonksiyonu
    def calculate_mcnemar(y_true, pred1, pred2):
        # b: Model 1 doğru, Model 2 yanlış
        # c: Model 1 yanlış, Model 2 doğru
        b = np.sum((pred1 == y_true) & (pred2 != y_true))
        c = np.sum((pred1 != y_true) & (pred2 == y_true))
        
        table = [[0, b], [c, 0]]
        # exact=True (Binom Testi) kullanıyoruz, örneklem küçük hatalarda hassas olsun diye.
        result = mcnemar(table, exact=True)
        return result.pvalue, b, c

    # 4. Karşılaştırmalar
    comparisons = [
        ("Lojistik Regresyon", "Random Forest", pred_lr, pred_rf),
        ("Lojistik Regresyon", "YSA", pred_lr, pred_ann),
        ("Random Forest", "YSA", pred_rf, pred_ann)
    ]

    results_data = []

    print("\n--- McNemar Testi Sonuçları ---")
    print(f"{'Kıyaslama':<35} | {'p-value':<10} | {'Sonuç'}")
    print("-" * 65)
    
    for name1, name2, p1, p2 in comparisons:
        p_val, b, c = calculate_mcnemar(y_test, p1, p2)
        
        if p_val < 0.05:
            conclusion = "ANLAMLI FARK VAR"
        else:
            conclusion = "Fark Yok (Benzer Başarı)"
            
        print(f"{name1} vs {name2:<20} | {p_val:.5f}    | {conclusion}")
        
        results_data.append({
            'Model 1': name1,
            'Model 2': name2,
            'p-value': p_val,
            'Sonuç': conclusion,
            'Model 1 Doğru/Model 2 Yanlış Sayısı': b,
            'Model 1 Yanlış/Model 2 Doğru Sayısı': c
        })

    # Sonuçları Kaydet
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    pd.DataFrame(results_data).to_csv('outputs/McNemar_Sonuclari.csv', index=False)
    print("\n[BAŞARILI] McNemar sonuçları 'outputs/McNemar_Sonuclari.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    run_mcnemar()