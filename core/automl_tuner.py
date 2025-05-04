# core/automl_tuner.py
import pandas as pd
import numpy as np
import tensorflow as tf
import optuna

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class LogClassifierAutoML:
    def __init__(self, csv_path, label_col='label'):
        self.csv_path = csv_path
        self.label_col = label_col
        self.max_words = 5000
        self.max_len = 100

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        df[self.label_col] = df[self.label_col].astype(str)
        texts = df["cleaned_log"].astype(str).tolist()
        labels = df[self.label_col].tolist()
        return texts, labels

    def preprocess(self, texts, labels):
        tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)  # 'normal' -> 0, 'failure' -> 1

        return padded, encoded_labels, tokenizer, label_encoder

    def build_model(self, trial):
        embedding_dim = trial.suggest_categorical("embedding_dim", [32, 64, 128])
        lstm_units = trial.suggest_categorical("lstm_units", [32, 64, 128])
        dropout_rate = trial.suggest_float("dropout", 0.2, 0.5)

        model = Sequential()
        model.add(Embedding(self.max_words, embedding_dim, input_length=self.max_len))
        model.add(LSTM(lstm_units))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def objective(self, trial):
        texts, labels = self.load_data()
        X, y, _, _ = self.preprocess(texts, labels)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = self.build_model(trial)

        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=10, batch_size=32, callbacks=[es], verbose=0)

        return max(history.history["val_accuracy"])

    def optimize(self, n_trials=10):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        print(f"[✅] Best trial accuracy: {study.best_trial.value:.4f}")
        print("[🏆] Best hyperparameters:", study.best_trial.params)


# Run this module
if __name__ == "__main__":
    automl = LogClassifierAutoML(csv_path="D:\\aish_test_app\\data\\parsed_mozilla_crash_logs.csv")
    automl.optimize(n_trials=10)
