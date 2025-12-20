from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import os
import webbrowser
from threading import Timer

app = Flask(__name__)
CORS(app)

# 1. Modelleri ve Encoderları Yükle
MODEL_PATH = 'models/best_model.pkl'
ENCODER_PATH = 'models/encoders.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    print("\n[HATA] Model dosyaları bulunamadı! Lütfen önce 'save_final_model.py' dosyasını çalıştırın.")
    exit()

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)

# 2. Türkçe Sözlük (Gelen veriyi modelin anlayacağı harf kodlarına çevirmek için)
TR_MAPPING = {
    'cap-shape': {'b': 'Çan', 'c': 'Konik', 'x': 'Dışbükey', 'f': 'Düz', 'k': 'Topuzlu', 's': 'Çukur'},
    'cap-surface': {'f': 'Lifli', 'g': 'Oluklu', 'y': 'Pullu', 's': 'Pürüzsüz'},
    'cap-color': {'n': 'Kahverengi', 'b': 'Devetüyü', 'c': 'Gri', 'g': 'Yeşil', 'r': 'Pembe', 'p': 'Mor', 'u': 'Mor', 'e': 'Kırmızı', 'w': 'Beyaz', 'y': 'Sarı'},
    'bruises': {'t': 'Var', 'f': 'Yok'},
    'odor': {'a': 'Badem', 'l': 'Anason', 'c': 'Krep', 'y': 'Balık', 'f': 'Pis', 'm': 'Küf', 'n': 'Yok', 'p': 'Keskin', 's': 'Baharatlı'},
    'gill-attachment': {'a': 'Bağlı', 'd': 'İnişli', 'f': 'Serbest', 'n': 'Çentikli'},
    'gill-spacing': {'c': 'Yakın', 'w': 'Kalabalık', 'd': 'Uzak'},
    'gill-size': {'b': 'Geniş', 'n': 'Dar'},
    'gill-color': {'k': 'Siyah', 'n': 'Kahverengi', 'b': 'Devetüyü', 'h': 'Çikolata', 'g': 'Gri', 'r': 'Yeşil', 'o': 'Turuncu', 'p': 'Pembe', 'u': 'Mor', 'e': 'Kırmızı', 'w': 'Beyaz', 'y': 'Sarı'},
    'stalk-shape': {'e': 'Genişleyen', 't': 'Daralan'},
    'stalk-root': {'b': 'Yumrulu', 'c': 'Kulüp', 'u': 'Fincan', 'e': 'Eşit', 'z': 'Köksü', 'r': 'Köklü', '?': 'Bilinmiyor'},
    'stalk-surface-above-ring': {'f': 'Lifli', 'y': 'Pullu', 'k': 'İpeksi', 's': 'Pürüzsüz'},
    'stalk-surface-below-ring': {'f': 'Lifli', 'y': 'Pullu', 'k': 'İpeksi', 's': 'Pürüzsüz'},
    'stalk-color-above-ring': {'n': 'Kahverengi', 'b': 'Devetüyü', 'c': 'Gri', 'g': 'Yeşil', 'r': 'Pembe', 'p': 'Mor', 'e': 'Kırmızı', 'w': 'Beyaz', 'y': 'Sarı'},
    'stalk-color-below-ring': {'n': 'Kahverengi', 'b': 'Devetüyü', 'c': 'Gri', 'g': 'Yeşil', 'r': 'Pembe', 'p': 'Mor', 'e': 'Kırmızı', 'w': 'Beyaz', 'y': 'Sarı'},
    'veil-color': {'n': 'Kahverengi', 'o': 'Turuncu', 'w': 'Beyaz', 'y': 'Sarı'},
    'ring-number': {'n': 'Yok', 'o': 'Bir', 't': 'İki'},
    'ring-type': {'c': 'Kırılgan', 'e': 'Alevlenen', 'f': 'Büyük', 'l': 'Gidici', 'n': 'Yok', 'p': 'Sarkık', 's': 'Kılıflı', 'z': 'Bölge'},
    'spore-print-color': {'k': 'Siyah', 'n': 'Kahverengi', 'b': 'Devetüyü', 'h': 'Çikolata', 'r': 'Yeşil', 'o': 'Turuncu', 'u': 'Mor', 'w': 'Beyaz', 'y': 'Sarı'},
    'population': {'a': 'Bol', 'c': 'Küme', 'n': 'Çok', 's': 'Dağınık', 'v': 'Birkaç', 'y': 'Yalnız'},
    'habitat': {'g': 'Çayır', 'l': 'Yapraklar', 'm': 'Çayır', 'p': 'Yollar', 'u': 'Kentsel', 'w': 'Atık', 'd': 'Orman'}
}

# 3. Rotalar (Ana sayfa ve Statik dosyalar)
@app.route('/')
def index():
    # Direkt index.html dosyasını gönderir
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    # CSS, JS ve resim dosyalarını gönderir
    return send_from_directory('static', path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        encoded_input = []
        features = list(TR_MAPPING.keys())
        
        for col in features:
            tr_val = data.get(col)
            # Türkçe -> Harf (Örn: 'Çan' -> 'b')
            reverse_map = {v: k for k, v in TR_MAPPING[col].items()}
            letter_val = reverse_map.get(tr_val)
            
            # Harf -> Sayısal (LabelEncoder)
            encoded_val = encoders[col].transform([letter_val])[0]
            encoded_input.append(encoded_val)
            
        # Tahmin ve Olasılık (Hocanın istediği ana metrikler)
        prediction = model.predict([encoded_input])[0]
        probabilities = model.predict_proba([encoded_input])[0]
        prob = float(probabilities[prediction]) * 100
        
        target_label = encoders['target'].inverse_transform([prediction])[0]
        res_text = "ZEHİRLİ (Poisonous)" if target_label == 'p' else "YENİLEBİLİR (Edible)"
        
        return jsonify({
            'result': res_text,
            'probability': f"{prob:.2f}",
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

# Tarayıcıyı otomatik açma fonksiyonu
def open_browser():
      webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    print("\n--- Sunucu Başlatılıyor ---")
    # 1 saniye sonra tarayıcıyı açar
    Timer(1, open_browser).start()
    # Flask sunucusunu çalıştır
    app.run(port=5000, debug=False)