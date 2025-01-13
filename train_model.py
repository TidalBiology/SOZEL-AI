import wikipediaapi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import load_model

# Wikipedia verisini çekme fonksiyonu
def get_wikipedia_data(page_name):
    # User-Agent ekleniyor
    wiki_wiki = wikipediaapi.Wikipedia('en')
    page = wiki_wiki.page(page_name)
    return page.text  # Sayfa metnini döndürüyoruz

# Modeli oluşturma fonksiyonu
def create_model():
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64),  # Embedding katmanı
        LSTM(128),  # LSTM katmanı
        Dense(1, activation='sigmoid')  # Çıkış katmanı
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wikipedia'dan veri çekme (örnek olarak "Python (programming_language)" sayfası kullanılıyor)
wikipedia_text = get_wikipedia_data('Python (programming_language)')

# Örnek etiketleme - burada metnin pozitif olduğunu varsayıyoruz
texts = [wikipedia_text]  # Bu örnekte sadece bir metin kullanıyoruz
labels = [1]  # 1, pozitif etiket

# Tokenization ve Padding işlemleri
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=10, padding='post')  # Max uzunluk 10

# Modeli oluşturma ve eğitme
model = create_model()
model.fit(np.array(padded_sequences), np.array(labels), epochs=10)

# Modeli kaydetme
model.save("model/my_model.h5")
print("Model başarıyla kaydedildi.")

# Tahmin yapmak için bir fonksiyon
def predict(input_data):
    model = load_model("model/my_model.h5")
    tokenizer = Tokenizer(num_words=10000)
    sequences = tokenizer.texts_to_sequences([input_data])
    padded = pad_sequences(sequences, maxlen=10, padding='post')
    prediction = model.predict(np.array(padded))
    return prediction[0][0]

# Test - Verilen bir metinle tahmin yapma
input_text = "Python is an interpreted high-level programming language."
print(f'Tahmin sonucu: {predict(input_text)}')
