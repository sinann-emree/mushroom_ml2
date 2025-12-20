import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def setup_directories():
    """Hocanın istediği sonuçlar için klasör yapısını hazırlar."""
    directories = [
        'outputs/hold_out',
        'outputs/k_fold',
        'data'
    ]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Klasör oluşturuldu: {directory}")

def load_and_analyze(file_path):
    # 1. Veriyi yükle
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Hata: {file_path} bulunamadı! Lütfen dosyanın 'data' klasöründe olduğundan emin ol.")
    
    df = pd.read_csv(file_path)
    
    print("--- Veri Seti İlk 5 Satır ---")
    print(df.head())
    
    # 2. Eksik Veri Kontrolü
    print("\n--- '?' İşareti (Eksik Veri) Kontrolü ---")
    for col in df.columns:
        count = (df[col] == '?').sum()
        if count > 0:
            print(f"{col} sütununda {count} adet '?' (bilinmeyen değer) var.")

    # Gereksiz Sütun Analizi: 'veil-type' sadece tek bir değer ('p') içerdiği için öğrenmeye katkısı yoktur.
    if 'veil-type' in df.columns:
        print("\n'veil-type' sütunu tek tip değer içerdiği için siliniyor...")
        df = df.drop(['veil-type'], axis=1)

    return df

def preprocess_data(df):
    # 3. Kategorik Verileri Sayısallaştırma (Label Encoding)
    df_encoded = df.copy()
    le = LabelEncoder()
    
    # Tüm sütunları sayısal hale getiriyoruz
    for col in df_encoded.columns:
        df_encoded[col] = le.fit_transform(df_encoded[col])
        
    print("\n--- Sayısallaştırma Tamamlandı (İlk 5 Satır) ---")
    print(df_encoded.head())
    
    # İşlenmiş veriyi sonraki adımlarda kullanmak üzere kaydedelim
    df_encoded.to_csv('data/mushrooms_processed.csv', index=False)
    print("\nİşlenmiş veri 'data/mushrooms_processed.csv' olarak kaydedildi.")
    
    return df_encoded

if __name__ == "__main__":
    # Klasörleri hazırla
    setup_directories()
    
    # Dosya yolunu güncelledik
    file_path = os.path.join('data', 'mushrooms.csv') 
    
    try:
        df = load_and_analyze(file_path)
        df_processed = preprocess_data(df)
        print("\n[BAŞARILI] Veri ön işleme adımı tamamlandı. Bir sonraki adıma geçebiliriz.")
    except Exception as e:
        print(f"\n[HATA] Bir sorun oluştu: {e}")