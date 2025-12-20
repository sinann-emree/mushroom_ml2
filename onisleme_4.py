import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def split_dataset(file_path):
    print("--- ADIM 5: TRAIN / TEST SPLIT (Veri Sızıntısını Önleme) ---\n")
    
    # 1. Veriyi Yükle
    df = pd.read_csv(file_path)
    print(f"Orijinal Veri Boyutu: {df.shape}")

    # --- ÖNCEKİ ADIMLARDA ALINAN KARARLARIN UYGULANMASI ---
    
    # Adım 2 Kararı: '?' -> 'Unknown'
    df = df.replace('?', 'Unknown')
    
    # Adım 2 ve Adım 4 Kararları: Gereksiz Sütunları Silme
    # veil-type (tek değer), gill-attachment (%97 aynı), veil-color (%97 aynı)
    cols_to_drop = ['veil-type', 'gill-attachment', 'veil-color']
    
    # Sadece veride mevcut olanları sil (Hata almamak için kontrol)
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"-> Silinen Sütunlar: {existing_drops}")

    # 2. X (Özellikler) ve y (Hedef) Ayrımı
    X = df.drop('class', axis=1)
    y = df['class']

    # 3. Stratified Train-Test Split (%80 Eğitim, %20 Test)
    # stratify=y -> Sınıf dağılımını (oranını) korur.
    print("\nVeri seti %80 Eğitim - %20 Test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Eğitim Seti Boyutu (X_train): {X_train.shape}")
    print(f"Test Seti Boyutu   (X_test):  {X_test.shape}")

    # 4. Dağılım Kontrolü (Hocaya göstermek için önemli)
    print("\n--- Sınıf Dağılımı Kontrolü (Stratify Başarılı mı?) ---")
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    
    print("Eğitim Seti Oranları:\n", train_dist)
    print("-" * 20)
    print("Test Seti Oranları:\n", test_dist)

    # 5. Dosyaları Kaydet (Encoding aşaması için ham halde saklıyoruz)
    # index=False önemli, yoksa her okuduğumuzda gereksiz index sütunu oluşur.
    X_train.to_csv('data/X_train_raw.csv', index=False)
    X_test.to_csv('data/X_test_raw.csv', index=False)
    y_train.to_csv('data/y_train_raw.csv', index=False)
    y_test.to_csv('data/y_test_raw.csv', index=False)

    print("\n[BAŞARILI]: Ham parçalar 'data/' klasörüne kaydedildi.")
    print("-> X_train_raw.csv, X_test_raw.csv, y_train_raw.csv, y_test_raw.csv")
    print("Bir sonraki adımda bu dosyalara ENCODING işlemi yapacağız.")

if __name__ == "__main__":
    file_path = 'data/mushrooms.csv'
    split_dataset(file_path)