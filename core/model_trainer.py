# core/model_trainer.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import joblib
import os

class FinalModelTrainer:
    def __init__(self, data_path, model_path='data/models/lstm_classifier.h5'):
        self.data_path = data_path
        self.model_path = model_path
        self.max_words = 5000
        self.max_len = 100

    def load_and_preprocess(self):
        df = pd.read_csv(self.data_path)
        df = df[df['label'].isin(['normal', 'failure'])]  # Ensure only binary
        texts = df['cleaned_log'].astype(str).tolist()
        labels = df['label'].tolist()

        tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        encoder = LabelEncoder()
        y = encoder.fit_transform(labels)

        # Save tokenizer & encoder for later use
        joblib.dump(tokenizer, 'data/models/tokenizer.pkl')
        joblib.dump(encoder, 'data/models/label_encoder.pkl')

        return padded, y

    def build_and_train_model(self, X, y):
        model = Sequential()
        model.add(Embedding(self.max_words, 64))  # from best config
        model.add(LSTM(64))
        model.add(Dropout(0.49))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='loss', patience=2)
        model.fit(X, y, epochs=10, batch_size=32, callbacks=[es])

        model.save(self.model_path)
        print(f"[✅] Model saved at {self.model_path}")

# Run it
if __name__ == "__main__":
    os.makedirs("data/models", exist_ok=True)
    trainer = FinalModelTrainer("D:\\aish_test_app\\data\\parsed_mozilla_crash_logs.csv")
    X, y = trainer.load_and_preprocess()
    trainer.build_and_train_model(X, y)