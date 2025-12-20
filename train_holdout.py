import pandas as pd
from sklearn.model_selection import train_test_split
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
X = df.drop('class', axis=1)
y = df['class']

# 2. Hold-out (%80 Eğitim, %20 Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "kNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB(),
    "SVM": SVC(probability=True),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = []

print("--- Klasik Modeller Eğitiliyor (Hold-out) ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Metrikleri hesapla
    res = calculate_metrics(y_test, y_pred, name)
    results.append(res)
    
    # Grafikleri kaydet
    save_confusion_matrix(y_test, y_pred, name, "hold_out")
    save_roc_curve(y_test, y_probs, name, "hold_out")
    print(f"{name} tamamlandı.")

# 3. Yapay Sinir Ağları (ANN/YSA)
print("\n--- YSA Eğitiliyor (Epoch >= 100) ---")
ysa_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

ysa_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim (Hocanın istediği gibi 100 epoch)
history = ysa_model.fit(X_train, y_train, epochs=100, batch_size=32, 
                    validation_split=0.1, verbose=0)

# YSA Tahminleri
ysa_probs = ysa_model.predict(X_test).flatten()
ysa_pred = (ysa_probs > 0.5).astype(int)

# YSA Metrik ve Grafikler
results.append(calculate_metrics(y_test, ysa_pred, "ANN_YSA"))
save_confusion_matrix(y_test, ysa_pred, "ANN_YSA", "hold_out")
save_roc_curve(y_test, ysa_probs, "ANN_YSA", "hold_out")
save_ysa_plot(history, "hold_out")

# Sonuçları Tablo Olarak Kaydet ve Göster
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/hold_out/performance_results.csv', index=False)
print("\n--- HOLD-OUT SONUÇLARI ---")
print(results_df)
print("\nGrafikler 'outputs/hold_out/' klasörüne kaydedildi.")