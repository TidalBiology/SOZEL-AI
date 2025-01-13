from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Flask uygulaması başlatma
app = Flask(__name__)

# Modeli oluşturma
def create_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Modeli eğitme fonksiyonu
def train_model():
    # Eğitim verisi
    texts = ["metin örneği 1", "metin örneği 2", "metin örneği 3"]
    labels = [0, 1, 0]

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')

    # Modeli oluşturma ve eğitme
    model = create_model()
    model.fit(np.array(padded_sequences), np.array(labels), epochs=10)

    # Modeli kaydetme
    model.save("model/my_model.h5")
    print("Model başarıyla kaydedildi.")

# Modeli yükleme fonksiyonu
def load_model():
    from tensorflow.keras.models import load_model
    if os.path.exists("model/my_model.h5"):
        return load_model("model/my_model.h5")
    else:
        print("Model bulunamadı, eğitim yapılması gerekiyor.")
        return None

# Ana sayfa
@app.route('/')
def home():
    return render_template('index.html')

# Tahmin yapma
@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    if model is None:
        return jsonify({'error': 'Model yüklenemedi.'})

    # Form verisini alma
    input_data = request.json.get('input_data', '')

    # Tokenizer yükleme ve veriyi işleme
    tokenizer = Tokenizer(num_words=10000)
    sequences = tokenizer.texts_to_sequences([input_data])
    padded = pad_sequences(sequences, maxlen=10, padding='post')

    # Tahmin yapma
    prediction = model.predict(np.array(padded))
    result = prediction[0][0]  # modelin çıktısı

    return jsonify({'prediction': result})

if __name__ == '__main__':
    # Eğer model yoksa eğitimi başlatıyoruz
    if not os.path.exists("model/my_model.h5"):
        train_model()
    
    app.run(debug=True)
