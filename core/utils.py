# core/utils.py
import sys
import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class Utils:
    """
    Utility functions for AISH-Test Framework.
    Loads the trained model, tokenizer, and label encoder.
    Provides methods for preprocessing logs and making predictions.
    """

    def __init__(self, models_base_dir="data/models"):
        """
        Initializes the Utils class by loading the trained model, tokenizer, and label encoder.

        Args:
            models_base_dir (str): The directory relative to project_root where models are stored.
                                   Defaults to "data/models".
        """
        self.model_path = os.path.join(project_root, models_base_dir, 'lstm_classifier.h5')
        self.tokenizer_path = os.path.join(project_root, models_base_dir, 'tokenizer.pkl')
        self.encoder_path = os.path.join(project_root, models_base_dir, 'label_encoder.pkl')

        # Check if model files exist
        for path in [self.model_path, self.tokenizer_path, self.encoder_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required model/tokenizer/encoder file not found at {path}. "
                    "Please ensure model_trainer.py has been run successfully."
                )
        
        print(f"[INFO] Utils: Loading Keras model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print(f"[INFO] Utils: Loading Tokenizer from: {self.tokenizer_path}")
        self.tokenizer = joblib.load(self.tokenizer_path)
        print(f"[INFO] Utils: Loading LabelEncoder from: {self.encoder_path}")
        self.encoder = joblib.load(self.encoder_path)
        
        # Max length should be consistent with training (from model_trainer.py)
        self.max_len = 150 # Assuming this was used in model_trainer.py, adjust if different
        print(f"[INFO] Utils initialized. Max sequence length set to: {self.max_len}")
        print(f"[INFO] Utils: Label encoder classes: {self.encoder.classes_}")


    def preprocess_logs(self, logs_list):
        """
        Preprocesses a list of raw log strings into model-ready padded sequences.
        Args:
            logs_list (list of str): A list of log messages (templates from Drain3).
        Returns:
            np.ndarray: Padded sequences.
        """
        if not isinstance(logs_list, list):
            raise TypeError("Input 'logs_list' must be a list of strings.")
        if not logs_list:
            print("[WARN] Utils.preprocess_logs: Received an empty list of logs.")
            return np.array([])

        print(f"[INFO] Utils: Preprocessing {len(logs_list)} logs for prediction...")
        sequences = self.tokenizer.texts_to_sequences(logs_list)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        # print(f"DEBUG Utils: Sample tokenized input for prediction (first log):\n{padded[0] if len(padded) > 0 else 'N/A'}")
        return padded

    def predict_logs(self, logs_list):
        """
        Predicts the probability and class for each log in the provided list.
        Args:
            logs_list (list of str): A list of log messages (templates from Drain3).
        Returns:
            list of tuples: Each tuple is (predicted_class_str, probability_float).
                            Returns an empty list if input is empty or preprocessing fails.
        """
        if not logs_list:
            print("[WARN] Utils.predict_logs: Received an empty list of logs for prediction.")
            return []
            
        padded_logs = self.preprocess_logs(logs_list)
        if padded_logs.size == 0: # Check if preprocessing returned an empty array
            print("[WARN] Utils.predict_logs: Preprocessing resulted in no data. Cannot predict.")
            return []

        print(f"[INFO] Utils: Making predictions with the loaded model...")
        # self.model.predict is expected to return probabilities for the positive class
        probabilities = self.model.predict(padded_logs).flatten() 
        
        # Thresholding probabilities to get binary predictions (0 or 1)
        # A threshold of 0.5 is standard for sigmoid output.
        binary_predictions = (probabilities > 0.5).astype(int)
        
        # Inverse transform binary predictions to original class labels (e.g., 'failure', 'normal')
        predicted_class_labels = self.encoder.inverse_transform(binary_predictions)
        
        results = list(zip(predicted_class_labels, probabilities))
        # print(f"DEBUG Utils: Sample prediction output (first log): {results[0] if results else 'N/A'}")
        return results

    @staticmethod
    def create_output_dataframe(logs_list, predictions_list):
        """
        Creates a Pandas DataFrame from logs and their predictions.
        Args:
            logs_list (list of str): The original log messages.
            predictions_list (list of tuples): List of (predicted_class_str, probability_float).
        Returns:
            pd.DataFrame: DataFrame with 'Log', 'Predicted Class', 'Prediction Confidence'.
        """
        if len(logs_list) != len(predictions_list):
            raise ValueError("Mismatch between the number of logs and predictions.")
            
        data_for_df = []
        for log_text, (pred_class, prob) in zip(logs_list, predictions_list):
            data_for_df.append({
                "Log": log_text,
                "Predicted Class": pred_class,
                "Prediction Confidence": f"{prob:.4f}" # Increased precision for confidence
            })
        return pd.DataFrame(data_for_df)

    @staticmethod
    def moving_average(values, window_size=5): # Renamed window to window_size for clarity
        """
        Computes a simple moving average over a list of values.
        Args:
            values (list or np.ndarray): List of numerical values.
            window_size (int): The size of the moving window.
        Returns:
            np.ndarray: Array of moving averages.
        """
        if not values or len(values) < window_size:
            return np.array([]) # Return empty if not enough data
        
        # Using np.convolve for efficient moving average calculation
        return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

if __name__ == '__main__':
    # This is a simple test case for Utils
    # Ensure that model_trainer.py has been run and models are in data/models/
    print("--- Testing Utils ---")
    try:
        utils_instance = Utils() # Uses default models_base_dir="data/models"
        
        # Sample logs (these should be Drain3 templates in a real scenario)
        sample_log_templates = [
            "Kernel panic: Unable to mount root filesystem", # Expected failure
            "Service restarted without manual intervention",   # Expected normal
            "Database connection timeout during query execution", # Expected failure
            "Heartbeat check passed for all nodes", # Expected normal
            "Crash in rendering engine: Null pointer dereference" # Expected failure
        ]
        
        print(f"\nTest logs for prediction: {sample_log_templates}")
        
        predictions_output = utils_instance.predict_logs(sample_log_templates)
        
        if predictions_output:
            print("\nPredictions:")
            for log, (p_class, p_prob) in zip(sample_log_templates, predictions_output):
                print(f"Log: \"{log[:50]}...\" -> Predicted: {p_class} (Confidence: {p_prob:.4f})")
            
            df_output = Utils.create_output_dataframe(sample_log_templates, predictions_output)
            print("\nOutput DataFrame:")
            print(df_output)
        else:
            print("\nNo predictions were made.")
            
    except FileNotFoundError as e:
        print(f"[❌] Test failed: {e}. Make sure models are trained and paths are correct.")
    except Exception as e:
        print(f"[❌] An unexpected error occurred during Utils test: {e}")
    print("--- Utils Test Finished ---")