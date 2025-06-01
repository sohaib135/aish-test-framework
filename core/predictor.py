# core/predictor.py
import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class LogPredictor:
    def __init__(self, models_base_dir="data/models", processed_data_path_relative="data/processed_logs.csv", 
                 text_col='parsed_log', label_col='label'):
        """
        Initializes the LogPredictor.

        Args:
            models_base_dir (str): Directory relative to project_root containing saved models.
            processed_data_path_relative (str): Path to the processed log data (CSV with templates),
                                                relative to project_root.
            text_col (str): Name of the column containing the log templates.
            label_col (str): Name of the column containing the labels.
        """
        self.model_path = os.path.join(project_root, models_base_dir, 'lstm_classifier.h5')
        self.tokenizer_path = os.path.join(project_root, models_base_dir, 'tokenizer.pkl')
        self.encoder_path = os.path.join(project_root, models_base_dir, 'label_encoder.pkl')
        self.processed_data_path = os.path.join(project_root, processed_data_path_relative)
        
        self.text_col = text_col
        self.label_col = label_col
        
        # Max length should be consistent with training (from model_trainer.py / utils.py)
        self.max_len = 150 # Adjust if different in your training configuration

        # Load models and preprocessors
        print(f"[INFO] Predictor: Loading Keras model from: {self.model_path}")
        self.model = load_model(self.model_path)
        print(f"[INFO] Predictor: Loading Tokenizer from: {self.tokenizer_path}")
        self.tokenizer = joblib.load(self.tokenizer_path)
        print(f"[INFO] Predictor: Loading LabelEncoder from: {self.encoder_path}")
        self.encoder = joblib.load(self.encoder_path)
        print(f"[INFO] Predictor: Label encoder classes: {self.encoder.classes_}")


    def load_evaluation_data(self):
        """
        Loads and preprocesses the evaluation data from processed_logs.csv.
        Uses the 'parsed_log' column which contains Drain3 templates.
        """
        print(f"[INFO] Predictor: Loading evaluation data from: {self.processed_data_path}")
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Evaluation data file not found: {self.processed_data_path}")
            
        df = pd.read_csv(self.processed_data_path)
        
        # Filter for 'normal' and 'failure' labels and handle NaNs
        df = df[df[self.label_col].isin(['normal', 'failure'])].copy()
        df.dropna(subset=[self.text_col], inplace=True)
        
        if df.empty:
            raise ValueError("No data available for evaluation after filtering. Check processed_logs.csv.")

        texts = df[self.text_col].astype(str).tolist() # Use the parsed_log (templates)
        
        # Transform labels using the loaded encoder
        # Ensure labels are strings before transform if they aren't already
        try:
            labels_encoded = self.encoder.transform(df[self.label_col].astype(str).tolist())
        except ValueError as e:
            unknown_labels = set(df[self.label_col].astype(str).tolist()) - set(self.encoder.classes_)
            raise ValueError(f"Labels in evaluation data contain unseen values: {unknown_labels}. Error: {e}")

        print(f"[INFO] Predictor: Preprocessing {len(texts)} log templates for evaluation...")
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded_sequences, labels_encoded

    def evaluate_model(self):
        """
        Evaluates the model on the loaded test data and prints a classification report
        and saves a confusion matrix.
        """
        try:
            X_eval, y_eval = self.load_evaluation_data()
        except (FileNotFoundError, ValueError) as e:
            print(f"[❌] Predictor: Error loading data for evaluation: {e}")
            return None

        if X_eval.size == 0:
            print("[❌] Predictor: No data to evaluate after preprocessing.")
            return None

        print(f"[INFO] Predictor: Evaluating model on {X_eval.shape[0]} samples...")
        # Get raw probabilities from the model
        probabilities = self.model.predict(X_eval).flatten()
        # Convert probabilities to binary class predictions (0 or 1)
        predicted_binary_classes = (probabilities > 0.5).astype(int)

        # Generate classification report
        # Ensure target_names are correctly ordered as per encoder.classes_
        report = classification_report(y_eval, predicted_binary_classes, target_names=self.encoder.classes_, zero_division=0)
        print("\n[📊] Classification Report on Evaluation Data:\n", report)

        # Generate and save confusion matrix
        cm = confusion_matrix(y_eval, predicted_binary_classes, labels=np.arange(len(self.encoder.classes_)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.encoder.classes_)
        
        plt.figure(figsize=(8, 6)) # Adjust figure size for better layout
        disp.plot(cmap='Blues', ax=plt.gca()) # Pass current axes to plot
        plt.title("Confusion Matrix – AISH-Test Model Evaluation")
        
        results_dir = os.path.join(project_root, "data", "results")
        os.makedirs(results_dir, exist_ok=True)
        fig_path = os.path.join(results_dir, "evaluation_confusion_matrix.png")
        
        try:
            plt.savefig(fig_path)
            print(f"[✅] Confusion matrix saved to {fig_path}")
        except Exception as e:
            print(f"[❌] Failed to save confusion matrix: {e}")
        finally:
            plt.close() # Close the plot to free memory

        return report, fig_path

if __name__ == "__main__":
    print(f"--- Running Log Predictor for Evaluation ---")
    # Define paths relative to the project root
    models_dir_relative = "data/models"
    processed_data_relative = "data/processed_logs.csv" # This should contain 'parsed_log' and 'label'

    try:
        predictor_instance = LogPredictor(
            models_base_dir=models_dir_relative,
            processed_data_path_relative=processed_data_relative 
        )
        predictor_instance.evaluate_model()
    except FileNotFoundError as e:
        print(f"[❌] Evaluation failed: {e}. Ensure models are trained and data paths are correct.")
    except ValueError as e:
        print(f"[❌] Evaluation failed due to data issue: {e}.")
    except Exception as e:
        print(f"[❌] An unexpected error occurred during evaluation: {e}")
    
    print(f"--- Log Predictor Evaluation Finished ---")