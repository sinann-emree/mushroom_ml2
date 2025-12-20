from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import webbrowser
from threading import Timer

app = Flask(__name__)

# Modelleri Yükle
print("Modeller yükleniyor...")
try:
    # 1. Eğitilmiş Model (Hold-out adımında kaydedilen Random Forest)
    model = joblib.load('models/best_model_holdout.pkl')
    
    # 2. Ön İşlemci (One-Hot Encoder vb. ayarları)
    preprocessor = joblib.load('models/preprocessor.joblib')
    
    # 3. Hedef Dönüştürücü (0->e, 1->p)
    label_encoder = joblib.load('models/label_encoder.joblib')
    
    print("[BAŞARILI] Tüm modeller yüklendi.")
except Exception as e:
    print(f"[HATA] Model yüklenemedi: {e}")
    print("Lütfen önce 'model_holdout.py' ve 'onisleme_6.py' çalıştırdığınızdan emin olun.")
    exit()

# Sütun İsimleri (Eğitim sırasıyla AYNI olmalı)
# Adım 5'te sildiğimiz (veil-type, gill-attachment, veil-color) HARİÇ diğerleri.
EXPECTED_COLS = [
    'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    'stalk-surface-above-ring', 'stalk-surface-below-ring', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Gelen veriyi DataFrame'e çevir
        # Kullanıcıdan gelmeyen sütunlar varsa (silinmiş olanlar vb.) hata almamak için dikkatli olmalıyız
        input_data = {col: [data.get(col)] for col in EXPECTED_COLS}
        input_df = pd.DataFrame(input_data)
        
        # 2. Veriyi Dönüştür (Pipeline Transform)
        # One-Hot Encoding işlemini eğitimdeki kurallarla uygular
        input_transformed = preprocessor.transform(input_df)
        
        # 3. Tahmin Yap (predict ve predict_proba)
        prediction_idx = model.predict(input_transformed)[0]
        probs = model.predict_proba(input_transformed)[0]
        probability = probs[prediction_idx] * 100
        
        # 4. Sonucu Çöz (0/1 -> e/p)
        pred_label = label_encoder.inverse_transform([prediction_idx])[0]
        
        # 5. Ekrana Basılacak Mesaj
        if pred_label == 'p':
            result_text = "ZEHİRLİ"
        else:
            result_text = "YENİLEBİLİR"
            
        return jsonify({
            'status': 'success',
            'prediction': result_text,
            'probability': f"{probability:.2f}"
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=5000, debug=False)