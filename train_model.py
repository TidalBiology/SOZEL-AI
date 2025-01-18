import wikipediaapi
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import json

# Wikipedia verisini çekme fonksiyonu
def get_wikipedia_data(page_name):
    """
    Wikipedia'dan verilen sayfa adını kullanarak içerik getirir.
    """
    user_agent = "SOZEL-AI (contact: your_email@example.com)"  # User-Agent bilgisi
    wiki_wiki = wikipediaapi.Wikipedia(language='en', user_agent=user_agent)
    page = wiki_wiki.page(page_name)
    if not page.exists():
        raise ValueError(f"'{page_name}' sayfası bulunamadı.")
    return page.text

# Model oluşturma fonksiyonu
def create_model():
    """
    LSTM tabanlı bir model oluşturur.
    """
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64, input_length=100),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Model eğitme fonksiyonu
def train_model(page_name):
    """
    Wikipedia'dan veri alır, model oluşturur ve eğitir.
    Model başarıyla eğitildiğinde model kaydedilir.
    """
    # Wikipedia'dan veri çekme
    print(f"'{page_name}' sayfasından veri çekiliyor...")
    wikipedia_text = get_wikipedia_data(page_name)

    # Eğitim verisi hazırlama
    texts = wikipedia_text.split('.')  # Cümle bazlı segmentasyon
    texts = [text.strip() for text in texts if len(text.strip()) > 10]  # Kısa cümleleri çıkar
    labels = [1] * len(texts)  # Tüm cümleleri pozitif olarak etiketle (örnek için)

    if not texts:
        raise ValueError("Wikipedia'dan alınan veri yeterli uzunlukta değil.")

    # Tokenizer ve Padding işlemleri
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    # Model oluşturma ve eğitme
    model = create_model()
    print("Model eğitiliyor...")
    model.fit(np.array(padded_sequences), np.array(labels), epochs=5, batch_size=32)

    # Modeli ve tokenizer'ı kaydetme
    os.makedirs("model", exist_ok=True)
    model.save("model/my_model.h5")
    with open("model/tokenizer.json", "w") as f:
        f.write(tokenizer.to_json())
    print("Model ve tokenizer başarıyla kaydedildi.")

# Tokenizer yükleme fonksiyonu
def load_tokenizer():
    """
    Daha önce kaydedilmiş tokenizer'ı yükler.
    """
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    tokenizer_path = "model/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Tokenizer dosyası bulunamadı. Lütfen modeli önce eğitin.")
    with open(tokenizer_path, "r") as f:
        tokenizer_data = f.read()
    return tokenizer_from_json(tokenizer_data)

# Tahmin yapma fonksiyonu
def predict(input_data):
    """
    Eğitimli model kullanarak verilen metin üzerinde tahmin yapar.
    """
    model_path = "model/my_model.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model dosyası bulunamadı. Lütfen önce modeli eğitin.")

    # Modeli ve tokenizer'ı yükleme
    print("Model yükleniyor...")
    model = load_model(model_path)
    tokenizer = load_tokenizer()

    # Tokenizer işlemleri
    sequences = tokenizer.texts_to_sequences([input_data])
    padded = pad_sequences(sequences, maxlen=100, padding='post')

    # Tahmin yapma
    print("Tahmin yapılıyor...")
    prediction = model.predict(np.array(padded))
    return prediction[0][0]

# Ana Çalışma Bloğu
if __name__ == "__main__":
    try:
        # Model eğitimi
        train_model("Python (programming_language)")
    except ValueError as e:
        print(f"Hata: {e}")

    # Tahmin yapma
    input_text = "Python is an interpreted high-level programming language."
    try:
        result = predict(input_text)
        print(f"Tahmin Sonucu: {result:.4f}")
    except FileNotFoundError as e:
        print(f"Hata: {e}")
