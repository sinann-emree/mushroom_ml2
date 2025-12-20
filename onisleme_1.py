import pandas as pd
import numpy as np
import os

def analyze_missing_values(file_path):
    print("--- ADIM 2: EKSİK VERİ VE TİP ANALİZİ ---\n")
    
    # 1. Veriyi Yükle
    df = pd.read_csv(file_path)
    print(f"Veri Seti Boyutu: {df.shape}")
    print("-" * 30)

    # 2. Veri Tiplerini Göster
    print("Veri Tipleri:\n")
    print(df.dtypes)
    print("-" * 30)

    # 3. Standart NaN Kontrolü
    print("Standart 'NaN' (Boş) Değer Sayıları:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    if df.isnull().sum().sum() == 0:
        print("-> Standart NaN değeri bulunamadı.")
    print("-" * 30)

    # 4. Gizli Eksik Veri ('?') Analizi
    # Mushroom veri setinde eksik veriler '?' ile kodlanmış olabilir.
    print("Gizli Eksik Veri ('?') Analizi:")
    question_mark_counts = {}
    
    for col in df.columns:
        # Sadece object (string) tipindeki sütunlarda arama yap
        if df[col].dtype == 'object':
            count = (df[col] == '?').sum()
            if count > 0:
                percentage = (count / len(df)) * 100
                question_mark_counts[col] = {'count': count, 'percent': percentage}
                print(f"-> '{col}' sütununda {count} adet '?' var. (Oran: %{percentage:.2f})")
    
    if not question_mark_counts:
        print("-> Veri setinde '?' işareti bulunamadı.")
    else:
        print("\n[YORUM]: Bu sütunlar için bir doldurma (imputation) veya silme stratejisi belirlemeliyiz.")

    print("-" * 30)
    
    # 5. Tekil Değer (Unique Value) Kontrolü
    # Gereksiz sütun tespiti için ön hazırlık
    print("Sütunlardaki Benzersiz Değer Sayıları (Cardinality):")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count}")
        if unique_count == 1:
            print(f"   [UYARI] '{col}' sütunu sadece 1 çeşit değer içeriyor! (Gereksiz Sütun)")

if __name__ == "__main__":
    file_path = 'data/mushrooms.csv'
    analyze_missing_values(file_path)