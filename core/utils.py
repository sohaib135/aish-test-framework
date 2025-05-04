# core/utils.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Utils:
    """
    Utility functions for AISH-Test Framework.
    """

    def __init__(self, model_path, tokenizer_path, encoder_path):
        self.model = load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.encoder = joblib.load(encoder_path)
        self.max_len = 100

    def preprocess_logs(self, logs):
        """
        Preprocess raw log list into model-ready padded sequences.
        """
        sequences = self.tokenizer.texts_to_sequences(logs)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return padded

    def predict_logs(self, logs):
        """
        Predict the probability and class for each log.
        Returns a list of tuples: (predicted_class, probability)
        """
        padded_logs = self.preprocess_logs(logs)
        probs = self.model.predict(padded_logs).flatten()
        predictions = (probs > 0.5).astype(int)
        classes = self.encoder.inverse_transform(predictions)
        return list(zip(classes, probs))

    @staticmethod
    def create_output_dataframe(logs, predictions):
        """
        Create a DataFrame for dashboard or report.
        """
        data = []
        for log, (pred_class, prob) in zip(logs, predictions):
            data.append({
                "Log": log,
                "Predicted Class": pred_class,
                "Prediction Confidence": f"{prob:.2f}"
            })
        return pd.DataFrame(data)

    @staticmethod
    def moving_average(values, window=5):
        """
        Compute simple moving average over a window.
        """
        if len(values) < window:
            return np.mean(values)
        return np.convolve(values, np.ones(window)/window, mode='valid')
