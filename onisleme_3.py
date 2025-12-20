import pandas as pd
import numpy as np
import os

def analyze_weak_features(file_path):
    print("--- ADIM 4: ZAYIF ÖZELLİK ANALİZİ VE SİLME ÖNERİSİ ---\n")
    
    # 1. Veriyi Yükle
    df = pd.read_csv(file_path)
    
    # Mevcut Durum (Önceki kararları uygula)
    if 'veil-type' in df.columns:
        df = df.drop('veil-type', axis=1)
    df = df.replace('?', 'Unknown')

    # 2. Zayıf Olduğu Tespit Edilen Özellikler (Adım 3 sonuçlarına göre < 0.02)
    # stalk-shape: 0.005, gill-attachment: 0.009, veil-color: 0.016, cap-surface: 0.019
    weak_features = ['stalk-shape', 'gill-attachment', 'veil-color', 'cap-surface']
    
    print(f"İncelenecek Zayıf Adaylar: {weak_features}\n")
    
    print(f"{'Özellik':<20} | {'Benzersiz Değer Sayısı':<22} | {'En Çok Geçen Değer (%)':<25}")
    print("-" * 75)
    
    features_to_drop = []
    
    for col in weak_features:
        unique_count = df[col].nunique()
        # En sık geçen değerin oranını bul (Dominantlık)
        top_value_ratio = df[col].value_counts(normalize=True).iloc[0] * 100
        
        print(f"{col:<20} | {unique_count:<22} | %{top_value_ratio:.2f}")
        
        # Karar Mekanizması:
        # Eğer bir özellik %95'ten fazla aynı değere sahipse, ayırt ediciliği çok düşüktür.
        if top_value_ratio > 90: 
            features_to_drop.append(col)
            
    print("-" * 75)
    
    print("\n[ÖNERİLEN AKSİYONLAR]:")
    if features_to_drop:
        print(f"1. Şu sütunlar verideki çeşitliliği çok az olduğu için SİLİNMELİDİR: {features_to_drop}")
        print("   -> Sebebi: Neredeyse tüm satırlarda aynı değere sahipler (Low Variance).")
    else:
        print("1. İstatistiksel olarak zayıf görünseler de veri çeşitliliği var, silmeyebiliriz.")

    # cap-surface özel yorumu (Sınırda kalmıştı: 0.019)
    if 'cap-surface' not in features_to_drop:
        print("2. 'cap-surface' skoru düşük olsa da çeşitlilik içerdiği için TUTULABİLİR.")
        
    print("\nSonraki adımda (ADIM 5) bu önerilen sütunları silerek Train/Test ayrımı yapacağız.")

if __name__ == "__main__":
    file_path = 'data/mushrooms.csv'
    analyze_weak_features(file_path)