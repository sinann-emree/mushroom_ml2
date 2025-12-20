import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf
from utils import calculate_metrics, save_confusion_matrix, save_roc_curve, save_ysa_plot

# 1. Veriyi Yükle
df = pd.read_csv('data/mushrooms_processed.csv')
X = df.drop('class', axis=1).values
y = df['class'].values

# 2. K-Fold Hazırlığı
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models_configs = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

all_fold_results = []

print("--- K-Fold Çapraz Doğrulama Başlıyor (k=5) ---")

fold_idx = 1
for train_index, test_index in kf.split(X, y):
    print(f"\nFold {fold_idx} işleniyor...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    for name, model in models_configs.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, name)
        metrics['Fold'] = fold_idx
        all_fold_results.append(metrics)
        
        if fold_idx == 1:
            save_confusion_matrix(y_test, y_pred, name, "k_fold")
            save_roc_curve(y_test, y_probs, name, "k_fold")
            
    # 3. YSA (ANN) - Düzeltilmiş Kısım
    ysa_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)), # Yeni Input tarzı
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    ysa_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # validation_split ekleyerek 'val_accuracy' hatasını çözdük
    history = ysa_model.fit(X_train, y_train, epochs=100, batch_size=32, 
                            validation_split=0.1, verbose=0)
    
    ysa_probs = ysa_model.predict(X_test, verbose=0).flatten()
    ysa_pred = (ysa_probs > 0.5).astype(int)
    
    ysa_metrics = calculate_metrics(y_test, ysa_pred, "ANN_YSA")
    ysa_metrics['Fold'] = fold_idx
    all_fold_results.append(ysa_metrics)
    
    if fold_idx == 1:
        save_confusion_matrix(y_test, ysa_pred, "ANN_YSA", "k_fold")
        save_roc_curve(y_test, ysa_probs, "ANN_YSA", "k_fold")
        save_ysa_plot(history, "k_fold")
        
    fold_idx += 1

# 4. Sonuçları birleştir ve Ortalamaları Al
results_df = pd.DataFrame(all_fold_results)
final_avg_results = results_df.groupby('Model').agg({
    'Accuracy': 'mean',
    'Sensitivity': 'mean',
    'Specificity': 'mean',
    'F1-Score': 'mean'
}).reset_index()

final_avg_results.to_csv('outputs/k_fold/kfold_final_results.csv', index=False)

print("\n--- K-FOLD ORTALAMA SONUÇLARI ---")
print(final_avg_results)