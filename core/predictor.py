# core/predictor.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class LogPredictor:
    def __init__(self, model_path, tokenizer_path, encoder_path, data_path):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.encoder = joblib.load(encoder_path)
        self.data_path = data_path
        self.max_len = 100

    def load_data(self):
        df = pd.read_csv(self.data_path)
        df = df[df['label'].isin(['normal', 'failure'])]
        texts = df['cleaned_log'].astype(str).tolist()
        labels = self.encoder.transform(df['label'].tolist())
        return texts, labels

    def evaluate(self):
        texts, labels = self.load_data()
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')

        X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.3, random_state=42)

        predictions = self.model.predict(X_test).flatten()
        predicted_classes = (predictions > 0.5).astype(int)

        report = classification_report(y_test, predicted_classes, target_names=self.encoder.classes_)
        print("[📊] Classification Report:\n", report)

        cm = confusion_matrix(y_test, predicted_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.encoder.classes_)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix – AISH-Test Model")
        os.makedirs("D:/aish_test_app/data/results", exist_ok=True)
        fig_path = "D:/aish_test_app/data/results/confusion_matrix.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"[✅] Confusion matrix saved to {fig_path}")
        return fig_path


# Run it directly
if __name__ == "__main__":
    predictor = LogPredictor(
        model_path="D:/aish_test_app/data/models/lstm_classifier.h5",
        tokenizer_path="D:/aish_test_app/data/models/tokenizer.pkl",
        encoder_path="D:/aish_test_app/data/models/label_encoder.pkl",
        data_path="D:/aish_test_app/data/parsed_mozilla_crash_logs.csv"
    )
    predictor.evaluate()
