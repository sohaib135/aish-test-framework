# core/automl_tuner.py
import pandas as pd
import numpy as np
import tensorflow as tf
import optuna
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class LogClassifierAutoML:
    def __init__(self, processed_csv_path, label_col='label', text_col='parsed_log'):
        self.processed_csv_path = processed_csv_path
        self.label_col = label_col
        self.text_col = text_col # Column containing Drain3 parsed templates
        self.max_words = 10000  # Increased vocabulary size
        self.max_len = 150     # Increased sequence length

        # Check if the CSV file exists
        if not os.path.exists(self.processed_csv_path):
            raise FileNotFoundError(
                f"Processed log file not found at {self.processed_csv_path}. "
                "Please run log_ingestor.py first to generate processed_logs.csv."
            )
        print(f"[INFO] AutoMLTuner initialized with data from: {self.processed_csv_path}")


    def load_data(self):
        print(f"[INFO] AutoMLTuner: Loading data from {self.processed_csv_path}")
        df = pd.read_csv(self.processed_csv_path)
        
        if self.text_col not in df.columns:
            raise ValueError(f"Text column '{self.text_col}' not found in {self.processed_csv_path}.")
        if self.label_col not in df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in {self.processed_csv_path}.")

        # Filter for only 'normal' and 'failure' labels
        df = df[df[self.label_col].isin(['normal', 'failure'])].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # Explicitly convert to string type
        df.loc[:, self.text_col] = df[self.text_col].astype(str)
        df.loc[:, self.label_col] = df[self.label_col].astype(str)

        texts = df[self.text_col].tolist()
        labels = df[self.label_col].tolist()
        
        print(f"[INFO] AutoMLTuner: Loaded {len(texts)} logs for tuning.")
        return texts, labels

    def preprocess_data(self, texts, labels):
        print("[INFO] AutoMLTuner: Preprocessing data...")
        tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        print(f"[INFO] AutoMLTuner: Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
        
        print(f"[INFO] AutoMLTuner: Data preprocessed. Shape of padded sequences: {padded_sequences.shape}")
        return padded_sequences, encoded_labels, tokenizer, label_encoder

    def build_model(self, trial, input_dim_tokenizer):
        # Hyperparameters to tune
        embedding_dim = trial.suggest_categorical("embedding_dim", [64, 128, 256])
        lstm_units = trial.suggest_categorical("lstm_units", [64, 128, 256])
        dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6, step=0.1) # Corrected parameter name
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = Sequential([
            Embedding(input_dim=input_dim_tokenizer, output_dim=embedding_dim, input_length=self.max_len),
            LSTM(lstm_units, return_sequences=False), # Only return last output if not stacking LSTMs
            Dropout(dropout_rate),
            Dense(64, activation='relu'), # Added an intermediate dense layer
            Dropout(dropout_rate / 2),    # Added another dropout
            Dense(1, activation='sigmoid') # Binary classification
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def objective(self, trial):
        print(f"\n--- Optuna Trial {trial.number} ---")
        texts, labels = self.load_data()
        # Important: Pass the actual tokenizer to build_model to get the correct vocab size
        # For the objective function, we re-preprocess each time to ensure fresh state if needed,
        # but the tokenizer's vocab size should be consistent.
        # A better approach for vocab_size would be to fit tokenizer once before study.optimize.
        # For simplicity here, we'll fit it and pass its effective vocab size.
        
        # Fit tokenizer once to get consistent vocab_size for all trials
        temp_tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        temp_tokenizer.fit_on_texts(texts)
        # Use len(tokenizer.word_index) + 1 for more accurate vocab size if num_words is just a cap
        effective_vocab_size = min(self.max_words, len(temp_tokenizer.word_index) + 1)


        padded_sequences, encoded_labels, _, _ = self.preprocess_data(texts, labels)
        
        # Stratified split is important for potentially imbalanced (though this dataset is balanced)
        X_train, X_val, y_train, y_val = train_test_split(
            padded_sequences, encoded_labels, test_size=0.25, random_state=42, stratify=encoded_labels
        )
        
        print(f"[INFO] AutoMLTuner Trial {trial.number}: Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

        model = self.build_model(trial, input_dim_tokenizer=effective_vocab_size) # Pass vocab size

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max', verbose=1)
        
        print(f"[INFO] AutoMLTuner Trial {trial.number}: Starting model training...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20, # Increased epochs
            batch_size=trial.suggest_categorical("batch_size", [32, 64]),
            callbacks=[early_stopping],
            verbose=0 # Set to 1 for more detailed Keras output per epoch
        )
        
        val_accuracy = np.max(history.history["val_accuracy"])
        print(f"[INFO] AutoMLTuner Trial {trial.number}: Validation accuracy = {val_accuracy:.4f}")
        return val_accuracy

    def optimize_hyperparameters(self, n_trials=20): # Increased default trials
        print(f"\n--- Starting Optuna Hyperparameter Optimization ({n_trials} trials) ---")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, timeout=1800) # Added a timeout (e.g., 30 minutes)

        print("\n--- Optuna Optimization Finished ---")
        print(f"[✅] Best trial number: {study.best_trial.number}")
        print(f"[✅] Best validation accuracy: {study.best_trial.value:.4f}")
        print("[🏆] Best hyperparameters found:")
        for key, value in study.best_trial.params.items():
            print(f"  - {key}: {value}")
        
        return study.best_trial.params, study.best_trial.value

if __name__ == "__main__":
    # Define project root and path to processed_logs.csv relative to it
    processed_logs_path = os.path.join(project_root, "data", "processed_logs.csv")

    print(f"--- Running AutoML Tuner ---")
    automl_tuner = LogClassifierAutoML(processed_csv_path=processed_logs_path)
    
    # Run the optimization
    best_params, best_accuracy = automl_tuner.optimize_hyperparameters(n_trials=10) # Start with fewer trials if time is a concern

    print("\n--- AutoML Tuner Summary ---")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")
    print("Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("You should use these hyperparameters in model_trainer.py")