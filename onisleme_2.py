import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

def analyze_relationships(file_path):
    print("--- ADIM 3: İLİŞKİ VE ÖNEM ANALİZİ (MUTUAL INFORMATION) ---\n")
    
    # Klasör kontrolü (Temizlik yaptığımız için tekrar oluşturalım)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # 1. Veriyi Yükle ve Temizle (Adım 2 kararlarına göre)
    df = pd.read_csv(file_path)
    
    # Karar 1: veil-type sil
    if 'veil-type' in df.columns:
        df = df.drop('veil-type', axis=1)
        print("-> 'veil-type' sütunu analizden çıkarıldı.")

    # Karar 2: '?' olanları 'Unknown' yap
    # (Bunu tüm veri setine uyguluyoruz, çünkü başka sütunlarda da çıkabilir)
    df = df.replace('?', 'Unknown')
    print("-> '?' değerleri 'Unknown' kategorisine dönüştürüldü.")

    # 2. Geçici Encoding (Mutual Info hesaplamak için sayısal veriye ihtiyaç var)
    # Bu encoding kalıcı değil, sadece analiz içindir.
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # 3. Mutual Information Hesaplama
    X = df_encoded.drop('class', axis=1)
    y = df_encoded['class']

    print("\nİlişki Skorları Hesaplanıyor (Lütfen bekleyin)...")
    importances = mutual_info_classif(X, y, discrete_features=True, random_state=42)
    
    # Sonuçları DataFrame'e dök
    feat_importances = pd.Series(importances, index=X.columns)
    feat_importances = feat_importances.sort_values(ascending=False)

    print("\n--- ÖZELLİK ÖNEM SKORLARI (Hedef Değişkene Etkisi) ---")
    print(feat_importances)

    # 4. Görselleştirme
    plt.figure(figsize=(12, 8))
    sns.barplot(x=feat_importances.values, y=feat_importances.index, palette='viridis')
    plt.title('Özelliklerin "Class" (Zehirli/Yenilebilir) ile İlişkisi (Mutual Info)')
    plt.xlabel('Mutual Information Skoru (Daha yüksek = Daha önemli)')
    plt.ylabel('Özellikler')
    plt.tight_layout()
    
    save_path = 'outputs/feature_importance.png'
    plt.savefig(save_path)
    print(f"\n[GRAFİK KAYDEDİLDİ]: {save_path}")
    print("\n[YORUM]: Skorları 0'a çok yakın olan özellikler model için gereksiz olabilir.")

if __name__ == "__main__":
    file_path = 'data/mushrooms.csv'
    analyze_relationships(file_path)