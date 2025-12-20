import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, roc_curve, auc

# Modeller
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# YSA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def run_kfold():
    print("--- 1.B) K-KAT ÇAPRAZ DOĞRULAMA (K-FOLD) YÖNTEMİ ---")
    
    output_dir = 'outputs/k_fold'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Verileri Birleştir (K-Fold tüm veri üzerinde yapılır)
    try:
        X_train = pd.read_csv('data/X_train_processed.csv')
        X_test = pd.read_csv('data/X_test_processed.csv')
        y_train = pd.read_csv('data/y_train_processed.csv')
        y_test = pd.read_csv('data/y_test_processed.csv')
        
        X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
        y = pd.concat([y_train, y_test], axis=0).values.ravel()
    except FileNotFoundError:
        print("[HATA] Veriler bulunamadı.")
        return

    print(f"Toplam Veri Boyutu: {X.shape}")

    # 2. K-Fold Ayarları (k=5)
    k = 5
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    models = {
        'Lojistik Regresyon': LogisticRegression(max_iter=1000),
        'Karar Agaci': DecisionTreeClassifier(),
        'K-En Yakin Komsuluk (KNN)': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True),
        'Rastgele Orman': RandomForestClassifier(n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    fold_results = []

    # 3. Klasik Modeller Döngüsü
    for name, model in models.items():
        print(f"\nModel: {name} (K-Fold Başlıyor)...")
        fold_accuracies = []
        
        # Her fold için
        for i, (train_index, val_index) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            y_prob = model.predict_proba(X_val_fold)[:, 1]
            
            # Metrikler
            acc = accuracy_score(y_val_fold, y_pred)
            sens = recall_score(y_val_fold, y_pred)
            f1 = f1_score(y_val_fold, y_pred)
            
            # Sadece İLK Fold için Grafik Kaydet (Hocaya örnek göstermek için yeterli)
            if i == 0:
                # CM
                cm = confusion_matrix(y_val_fold, y_pred)
                plt.figure(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
                plt.title(f'K-Fold (Fold 1) CM - {name}')
                plt.savefig(f'{output_dir}/CM_{name.replace(" ", "_")}.png')
                plt.close()
                
                # ROC
                fpr, tpr, _ = roc_curve(y_val_fold, y_prob)
                roc_auc = auc(fpr, tpr)
                plt.figure()
                plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.title(f'K-Fold (Fold 1) ROC - {name}')
                plt.legend()
                plt.savefig(f'{output_dir}/ROC_{name.replace(" ", "_")}.png')
                plt.close()
            
            fold_accuracies.append(acc)
        
        # Ortalamayı kaydet
        avg_acc = np.mean(fold_accuracies)
        print(f"-> {name} Ortalama Başarı: {avg_acc:.4f}")
        fold_results.append({'Model': name, 'Ortalama Accuracy': avg_acc})

    # 4. YSA için K-Fold
    print("\nModel: YSA (K-Fold Başlıyor)...")
    ysa_accuracies = []
    
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        model_ann = Sequential([
            Dense(32, activation='relu', input_shape=(X.shape[1],)),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Epoch 100
        history = model_ann.fit(X_train_fold, y_train_fold, epochs=100, batch_size=32, verbose=0)
        
        loss, acc = model_ann.evaluate(X_val_fold, y_val_fold, verbose=0)
        ysa_accuracies.append(acc)
        
        if i == 0:
            # YSA Kayıp Grafiği (Fold 1)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'])
            plt.title('YSA Accuracy (Fold 1)')
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'])
            plt.title('YSA Loss (Fold 1)')
            plt.savefig(f'{output_dir}/YSA_Grafikleri.png')
            plt.close()

    avg_ysa = np.mean(ysa_accuracies)
    print(f"-> YSA Ortalama Başarı: {avg_ysa:.4f}")
    fold_results.append({'Model': 'YSA', 'Ortalama Accuracy': avg_ysa})

    # Sonuçları Kaydet
    pd.DataFrame(fold_results).to_csv(f'{output_dir}/KFold_Sonuclari.csv', index=False)
    print("\n[BAŞARILI] K-Fold sonuçları 'outputs/k_fold' klasörüne kaydedildi.")

if __name__ == "__main__":
    run_kfold()