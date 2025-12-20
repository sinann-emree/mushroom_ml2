import pandas as pd
import numpy as np

def final_check():
    print("--- ADIM 7: SON KONTROL (Sanity Check) ---\n")
    
    # 1. İşlenmiş Verileri Yükle
    try:
        X_train = pd.read_csv('data/X_train_processed.csv')
        X_test = pd.read_csv('data/X_test_processed.csv')
        y_train = pd.read_csv('data/y_train_processed.csv')
        y_test = pd.read_csv('data/y_test_processed.csv')
    except FileNotFoundError:
        print("[HATA] İşlenmiş veriler bulunamadı. Önce onisleme_6.py çalıştırın.")
        return

    # 2. Boyut Kontrolleri
    print(f"Eğitim Seti: {X_train.shape} (Satır, Sütun)")
    print(f"Test Seti:   {X_test.shape} (Satır, Sütun)")
    
    if X_train.shape[1] != X_test.shape[1]:
        print("\n[KRİTİK HATA]: Eğitim ve Test setindeki sütun sayıları EŞİT DEĞİL!")
        print("Bu durum modelin test aşamasında çökmesine neden olur.")
    else:
        print("-> Sütun sayıları eşit ve uyumlu. (Başarılı)")

    # 3. Eksik Veri Kontrolü (Sıfır olmalı)
    total_nan = X_train.isnull().sum().sum() + X_test.isnull().sum().sum()
    print(f"\nToplam Eksik Değer (NaN) Sayısı: {total_nan}")
    if total_nan == 0:
        print("-> Veri seti tertemiz. (Başarılı)")
    else:
        print("-> [UYARI] Veride hala eksik değerler var!")

    # 4. Veri Tipi Kontrolü
    # Tüm sütunlar sayısal (int veya float) olmalı
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"\n[HATA] Sayısal olmayan sütunlar var: {non_numeric}")
    else:
        print("\n-> Tüm veriler sayısal formatta. (Başarılı)")

    # 5. Sınıf Dağılımı Son Kontrol
    print("\nSon Sınıf Dağılımı (Eğitim):")
    print(y_train['class'].value_counts(normalize=True))

    print("-" * 40)
    print("TEBRİKLER! Veri Ön İşleme Süreci Tamamlandı.")
    print("Artık 'train_holdout.py' gibi modelleme kodlarına geçebiliriz.")
    print("-" * 40)

if __name__ == "__main__":
    final_check()