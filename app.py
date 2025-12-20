from flask import Flask, request, jsonify, render_template
import joblib
import os
import webbrowser
from threading import Timer

app = Flask(__name__)

# Modelleri yükle
model = joblib.load('models/best_model.pkl')
encoders = joblib.load('models/encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_row = []
        
        # Gelen veriyi (harfleri) sayısal değerlere çevir
        for col in encoders.keys():
            if col == 'class': continue # Hedef değişkeni atla
            
            letter = data.get(col)
            # Unseen label hatasını engellemek için kontrol
            if letter in encoders[col].classes_:
                val = encoders[col].transform([letter])[0]
            else:
                # Eğer veride olmayan bir harf gelirse en sık görülen sınıfı ata
                val = 0 
            input_row.append(val)
            
        prediction = model.predict([input_row])[0]
        probabilities = model.predict_proba([input_row])[0]
        
        # Sonuç Metni
        target_name = encoders['class'].inverse_transform([prediction])[0]
        turkce_sonuc = "ZEHİRLİ" if target_name == 'p' else "YENİLEBİLİR"
        olasilik = f"{probabilities[prediction] * 100:.2f}"
        
        return jsonify({'status': 'success', 'class': turkce_sonuc, 'prob': olasilik})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    Timer(1, lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    app.run(port=5000, debug=False)