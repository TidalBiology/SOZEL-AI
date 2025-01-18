from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json

# Flask uygulaması başlatma
app = Flask(__name__)

# Modeli oluşturma fonksiyonu
def create_model():
    """
    LSTM tabanlı bir metin üretim modeli oluşturur.
    """
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64),
        LSTM(128, return_sequences=True),
        LSTM(128),
        Dense(10000, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Modeli eğitme fonksiyonu
def train_model():
    """
    Örnek verilerle modeli eğitir ve model dosyasını kaydeder.
    """
    # Eğitim verisi
    texts = [
        "Merhaba dünya. Bu bir örnek cümledir.",
        "Python ile yapay zeka geliştirmek çok eğlenceli.",
        "Flask ile web uygulamaları yapmak mümkündür."
    ]

    # Tokenizer işlemleri
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=20, padding='post')

    # Hedef verisi oluşturma
    X = padded_sequences[:, :-1]
    y = padded_sequences[:, 1:]

    # Model oluşturma ve eğitme
    model = create_model()
    print("Model eğitiliyor...")
    model.fit(X, y, epochs=50, batch_size=16)

    # Modeli ve tokenizer'ı kaydetme
    os.makedirs("model", exist_ok=True)
    model.save("model/my_model.h5")

    tokenizer_path = "model/tokenizer.json"
    with open(tokenizer_path, "w") as f:
        f.write(tokenizer.to_json())

    print("Model ve tokenizer başarıyla kaydedildi.")

# Eğitimli modeli ve tokenizer'ı yükleme fonksiyonu
def load_trained_model_and_tokenizer():
    """
    Daha önce eğitilmiş modeli ve tokenizer'ı yükler.
    """
    model_path = "model/my_model.h5"
    tokenizer_path = "model/tokenizer.json"

    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        print("Model ve tokenizer yükleniyor...")
        model = load_model(model_path)
        with open(tokenizer_path, "r") as f:
            tokenizer_json = json.load(f)  # JSON verisini yükleyin
            tokenizer = Tokenizer()  # Tokenizer nesnesini oluşturuyoruz
            tokenizer.word_index = tokenizer_json['word_index']  # JSON'dan word_index verisini yükleyin

        return model, tokenizer
    else:
        print("Model veya tokenizer bulunamadı. Lütfen önce eğitin.")
        return None, None


# Ana sayfa
@app.route('/')
def home():
    """
    Uygulamanın ana sayfasını döner.
    """
    return render_template('index.html')

# Tahmin yapma API'si
@app.route('/predict', methods=['POST'])
def predict():
    """
    Kullanıcıdan gelen veriye göre tahmin yapar.
    """
    model, tokenizer = load_trained_model_and_tokenizer()
    if model is None or tokenizer is None:
        return jsonify({'error': 'Model veya tokenizer yüklenemedi. Lütfen önce eğitin.'})

    # Kullanıcı girişini al
    input_data = request.json.get('input_data', '')

    if not input_data:
        return jsonify({'error': 'Geçersiz giriş. Lütfen metin sağlayın.'})

    # Tokenizer ve veriyi işleme
    input_sequence = tokenizer.texts_to_sequences([input_data])
    padded_sequence = pad_sequences(input_sequence, maxlen=19, padding='post')

    # Tahmin yapma
    prediction = model.predict(padded_sequence)
    predicted_index = np.argmax(prediction[0])
    predicted_word = tokenizer.index_word.get(predicted_index, '')

    return jsonify({'generated_word': predicted_word})

if __name__ == '__main__':
    """
    Uygulamayı başlatır. Eğer model mevcut değilse önce eğitir.
    """
    if not os.path.exists("model/my_model.h5"):
        print("Model bulunamadı, eğitim başlatılıyor...")
        train_model()

    app.run(debug=True)
