import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Sklearn Modelleri
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Metrikler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, auc

# YSA (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def run_holdout():
    print("--- 1.A) DIŞARIDA TUTMA (HOLD-OUT) YÖNTEMİ ---")
    
    # 1. Klasörleri Kontrol Et
    output_dir = 'outputs/hold_out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. İşlenmiş Verileri Yükle (Ön işlemeden gelenler)
    try:
        X_train = pd.read_csv('data/X_train_processed.csv')
        X_test = pd.read_csv('data/X_test_processed.csv')
        y_train = pd.read_csv('data/y_train_processed.csv').values.ravel()
        y_test = pd.read_csv('data/y_test_processed.csv').values.ravel()
    except FileNotFoundError:
        print("[HATA] Veri dosyaları yok. Lütfen önce onisleme kodlarını çalıştırın.")
        return

    print(f"Eğitim Verisi: {X_train.shape}, Test Verisi: {X_test.shape}")

    # 3. Klasik Modellerin Tanımlanması (5 Sınıflandırıcı + 2 Topluluk)
    models = {
        'Lojistik Regresyon': LogisticRegression(max_iter=1000),
        'Karar Agaci': DecisionTreeClassifier(),
        'K-En Yakin Komsuluk (KNN)': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'Destek Vektor Makineleri (SVM)': SVC(probability=True), # ROC için probability=True şart
        'Rastgele Orman (Random Forest)': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    results = []

    # 4. Klasik Modeller Döngüsü
    for name, model in models.items():
        print(f"\nModel Eğitiliyor: {name}...")
        model.fit(X_train, y_train)
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] # ROC için olasılık
        
        # a) Karışıklık Matrisi (Confusion Matrix)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Karışıklık Matrisi - {name}')
        plt.xlabel('Tahmin Edilen')
        plt.ylabel('Gerçek')
        plt.savefig(f'{output_dir}/CM_{name.replace(" ", "_")}.png')
        plt.close()
        
        # b) Metrikler (Doğruluk, Duyarlılık, Özgüllük, F1)
        tn, fp, fn, tp = cm.ravel()
        acc = accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred) # Duyarlılık
        spec = tn / (tn + fp) if (tn+fp) > 0 else 0 # Özgüllük
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'Doğruluk (Accuracy)': acc,
            'Duyarlılık (Sensitivity)': sens,
            'Özgüllük (Specificity)': spec,
            'F1-Skoru': f1
        })
        
        # c) ROC Eğrisi
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Eğrisi - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_dir}/ROC_{name.replace(" ", "_")}.png')
        plt.close()

        # UI için En İyi Modeli Kaydet (Örn: Random Forest)
        if name == 'Rastgele Orman (Random Forest)':
            joblib.dump(model, 'models/best_model_holdout.pkl')

    # 5. Yapay Sinir Ağları (YSA / ANN)
    print("\nModel Eğitiliyor: Yapay Sinir Ağları (YSA)...")
    model_ann = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Hoca Şartı: Epoch >= 100
    history = model_ann.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)
    
    # YSA Tahminleri
    y_prob_ann = model_ann.predict(X_test).ravel()
    y_pred_ann = (y_prob_ann > 0.5).astype(int)
    
    # YSA Metrikleri
    cm_ann = confusion_matrix(y_test, y_pred_ann)
    tn, fp, fn, tp = cm_ann.ravel()
    results.append({
        'Model': 'Yapay Sinir Aglari (YSA)',
        'Doğruluk (Accuracy)': accuracy_score(y_test, y_pred_ann),
        'Duyarlılık (Sensitivity)': recall_score(y_test, y_pred_ann),
        'Özgüllük (Specificity)': tn / (tn + fp),
        'F1-Skoru': f1_score(y_test, y_pred_ann)
    })
    
    # YSA Grafikleri
    # 1. Eğitim ve Kayıp Grafiği (Accuracy & Loss)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
    plt.title('YSA Model Başarısı (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('YSA Model Kaybı (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f'{output_dir}/YSA_Egitim_Grafikleri.png')
    plt.close()
    
    # 2. YSA CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title('Karışıklık Matrisi - YSA')
    plt.savefig(f'{output_dir}/CM_YSA.png')
    plt.close()
    
    # 3. YSA ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob_ann)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Eğrisi - YSA')
    plt.legend()
    plt.savefig(f'{output_dir}/ROC_YSA.png')
    plt.close()

    # 6. Sonuç Tablosunu Kaydet ve Göster
    df_results = pd.DataFrame(results)
    df_results.to_csv(f'{output_dir}/Holdout_Sonuclari.csv', index=False)
    
    print("\n" + "="*50)
    print("HOLD-OUT YÖNTEMİ SONUÇ TABLOSU")
    print("="*50)
    print(df_results)
    print("\n[BAŞARILI] Tüm grafikler ve tablolar 'outputs/hold_out' klasörüne kaydedildi.")

if __name__ == "__main__":
    run_holdout()