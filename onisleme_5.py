import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

def encode_data():
    print("--- ADIM 6: ENCODING (Kategorik -> Sayısal Dönüşüm) ---\n")
    
    # 1. Ham Parçaları Yükle (Adım 5 çıktıları)
    try:
        X_train = pd.read_csv('data/X_train_raw.csv')
        X_test = pd.read_csv('data/X_test_raw.csv')
        y_train = pd.read_csv('data/y_train_raw.csv').values.ravel() # Tek boyutlu array yap
        y_test = pd.read_csv('data/y_test_raw.csv').values.ravel()
    except FileNotFoundError:
        print("[HATA] Dosyalar bulunamadı. Lütfen önce onisleme_5.py çalıştırın.")
        return

    print(f"Ham Eğitim Verisi Boyutu: {X_train.shape}")
    
    # 2. X için One-Hot Encoding Hazırlığı
    # Tüm sütunlarımız kategorik olduğu için hepsini seçiyoruz.
    categorical_cols = X_train.columns.tolist()
    
    # ColumnTransformer Tanımla
    # handle_unknown='ignore': Test setinde eğitimde hiç görmediği bir kategori çıkarsa hata verme, 0 bas.
    # sparse_output=False: Çıktıyı sıkıştırılmış matris değil, normal tablo olarak ver (görselleştirmek için).
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough' # İşlenmeyen sütun varsa olduğu gibi bırak
    )
    
    print("\nEncoding işlemi başlıyor...")
    
    # 3. Fit ve Transform İşlemleri
    # DİKKAT: Sadece X_train üzerinde 'fit' yapıyoruz!
    X_train_encoded = preprocessor.fit_transform(X_train)
    # X_test üzerinde sadece 'transform' yapıyoruz!
    X_test_encoded = preprocessor.transform(X_test)
    
    # Sütun isimlerini geri kazan (One-Hot sonrası feature isimleri değişir)
    feature_names = preprocessor.get_feature_names_out(categorical_cols)
    
    # DataFrame'e dönüştür
    X_train_df = pd.DataFrame(X_train_encoded, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_encoded, columns=feature_names)
    
    print(f"One-Hot Encoding Sonrası Eğitim Boyutu: {X_train_df.shape}")
    print(f"-> Sütun sayısı {X_train.shape[1]}'den {X_train_df.shape[1]}'e çıktı. (Genişleme normal)")

    # 4. Y (Target) için Label Encoding
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"\nHedef Değişken Kodlaması: {le.classes_} -> [0, 1]")
    
    # 5. Dosyaları ve Encoderları Kaydet
    # İşlenmiş verileri kaydet
    X_train_df.to_csv('data/X_train_processed.csv', index=False)
    X_test_df.to_csv('data/X_test_processed.csv', index=False)
    
    # y verilerini DataFrame yaparak kaydedelim
    pd.DataFrame(y_train_encoded, columns=['class']).to_csv('data/y_train_processed.csv', index=False)
    pd.DataFrame(y_test_encoded, columns=['class']).to_csv('data/y_test_processed.csv', index=False)
    
    # Pipeline nesnelerini kaydet (İleride yeni veri gelirse kullanmak için şart!)
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    
    print("\n[BAŞARILI]: Tüm işlenmiş veriler 'data/' klasörüne kaydedildi.")
    print("[BAŞARILI]: Encoder nesneleri 'models/' klasörüne kaydedildi.")
    print("Artık model eğitimine (ADIM 7) geçmeye tam anlamıyla hazırız.")

if __name__ == "__main__":
    encode_data()