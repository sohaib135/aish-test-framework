# core/model_trainer.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

import joblib
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class FinalModelTrainer:
    def __init__(self, processed_data_path, models_dir='data/models', text_col='parsed_log', label_col='label'):
        self.processed_data_path = processed_data_path
        self.models_dir = models_dir # Directory to save models
        self.model_save_path = os.path.join(project_root, self.models_dir, 'lstm_classifier.h5')
        self.tokenizer_save_path = os.path.join(project_root, self.models_dir, 'tokenizer.pkl')
        self.encoder_save_path = os.path.join(project_root, self.models_dir, 'label_encoder.pkl')
        
        self.text_col = text_col
        self.label_col = label_col

        # Hyperparameters (Update these with your AutoML results or thesis optimal values)
        # Using thesis-reported "optimal" values from AutoML as a better default:
        # Embedding Dim 64, LSTM Units 128, Dropout Rate 0.205
        self.max_words = 10000  # Vocabulary size (consider tuning)
        self.max_len = 150     # Max sequence length (consider tuning)
        self.embedding_dim = 64 # From thesis optimal (AutoML)
        self.lstm_units = 128   # From thesis optimal (AutoML)
        self.dropout_rate = 0.30 # From thesis optimal (AutoML)
        self.batch_size = 32    # Common default, can be tuned
        self.epochs = 50        # Increased epochs
        self.learning_rate = 0.00018516260649192547 # Common default, can be tuned

        # Ensure models directory exists
        os.makedirs(os.path.join(project_root, self.models_dir), exist_ok=True)
        
        print(f"[INFO] ModelTrainer initialized. Models will be saved in: {os.path.join(project_root, self.models_dir)}")
        print(f"[INFO] Using Hyperparameters: max_words={self.max_words}, max_len={self.max_len}, "
              f"embedding_dim={self.embedding_dim}, lstm_units={self.lstm_units}, dropout_rate={self.dropout_rate}, "
              f"batch_size={self.batch_size}, epochs={self.epochs}, learning_rate={self.learning_rate}")


    def load_and_preprocess_data(self):
        print(f"[INFO] Loading and preprocessing data from: {self.processed_data_path}")
        if not os.path.exists(self.processed_data_path):
            raise FileNotFoundError(f"Processed data file not found: {self.processed_data_path}. Run log_ingestor.py first.")
            
        df = pd.read_csv(self.processed_data_path)
        
        # Filter for 'normal' and 'failure' labels only and handle NaN in text_col
        df = df[df[self.label_col].isin(['normal', 'failure'])].copy()
        df.dropna(subset=[self.text_col], inplace=True) # Drop rows where text_col is NaN
        
        texts = df[self.text_col].astype(str).tolist()
        labels = df[self.label_col].tolist()

        if not texts:
            raise ValueError("No text data found after filtering and cleaning. Check processed_logs.csv.")

        print(f"[INFO] Fitting Tokenizer on {len(texts)} log templates...")
        tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        # Determine effective vocabulary size for the Embedding layer
        self.effective_vocab_size = min(self.max_words, len(tokenizer.word_index) + 1)
        print(f"[INFO] Effective vocabulary size for Embedding layer: {self.effective_vocab_size}")


        print(f"[INFO] Encoding labels...")
        encoder = LabelEncoder()
        encoded_labels = encoder.fit_transform(labels)
        print(f"[INFO] Label mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
        # E.g., {'failure': 0, 'normal': 1} or vice-versa. This is important for interpreting predictions.

        # Save tokenizer & encoder
        joblib.dump(tokenizer, self.tokenizer_save_path)
        joblib.dump(encoder, self.encoder_save_path)
        print(f"[✅] Tokenizer saved to {self.tokenizer_save_path}")
        print(f"[✅] LabelEncoder saved to {self.encoder_save_path}")

        return padded_sequences, encoded_labels, encoder.classes_

    def build_and_train_model(self, X_data, y_data, class_labels):
        print("[INFO] Splitting data into training and validation sets (75/25)...")
        # Stratify helps maintain original class proportions in splits
        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.25, random_state=42, stratify=y_data)

        print(f"[INFO] Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")
        print(f"[INFO] Class distribution in training labels (y_train): {np.unique(y_train, return_counts=True)}")
        print(f"[INFO] Class distribution in validation labels (y_val): {np.unique(y_val, return_counts=True)}")
        
        # Calculate class weights (though your dataset is balanced, this is good practice)
        # This assigns more weight to the under-represented class during training
        class_weights_values = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train), # Use unique classes present in y_train
            y=y_train
        )
        # Keras expects a dictionary mapping class indices to weights
        class_weights_dict = dict(enumerate(class_weights_values))
        print(f"[INFO] Calculated class weights for training: {class_weights_dict}")


        print("[INFO] Building Keras LSTM model...")
        model = Sequential([
            Embedding(input_dim=self.effective_vocab_size, output_dim=self.embedding_dim, input_length=self.max_len, mask_zero=True), # mask_zero=True is good for padded sequences
            Bidirectional(LSTM(self.lstm_units, return_sequences=False)), # Using Bidirectional LSTM
            Dropout(self.dropout_rate),
            Dense(64, activation='relu'), # Intermediate dense layer
            Dropout(self.dropout_rate / 2),
            Dense(1, activation='sigmoid') # Output layer for binary classification
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        model.summary()

        # Callbacks
        # Monitor val_accuracy for early stopping, patience increased
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
        # Reduce learning rate if val_loss plateaus
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)

        print("[INFO] Starting model training...")
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights_dict, # Apply class weights
            verbose=1 # Show progress for each epoch
        )

        print("[INFO] Model training completed.")
        model.save(self.model_save_path)
        print(f"[✅] Model saved to {self.model_save_path}")

        # Evaluate on validation set
        loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
        print(f"\n--- Validation Set Performance ---")
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")
        
        return model, history

if __name__ == "__main__":
    # Define project root and path to processed_logs.csv relative to it
    processed_logs_file_path = os.path.join(project_root, "data", "processed_logs.csv")
    models_directory = os.path.join(project_root, "data", "models")

    print(f"--- Running Final Model Trainer ---")
    trainer = FinalModelTrainer(
        processed_data_path=processed_logs_file_path,
        models_dir="data/models" # This will be joined with project_root inside the class
    )
    
    # You would typically get these from automl_tuner.py output
    # For now, they are set as defaults in __init__ based on your thesis.
    # If you re-run automl_tuner.py and get better params, update them in trainer's __init__ or pass them here.
    # Example:
    # trainer.embedding_dim = best_params_from_automl['embedding_dim'] 
    # trainer.lstm_units = best_params_from_automl['lstm_units']
    # etc.

    try:
        X_data_padded, y_data_encoded, class_names = trainer.load_and_preprocess_data()
        if X_data_padded is not None and y_data_encoded is not None:
            trained_model, training_history = trainer.build_and_train_model(X_data_padded, y_data_encoded, class_names)
            print("\n[INFO] Training history (validation accuracy per epoch):")
            for epoch, acc in enumerate(training_history.history['val_accuracy']):
                print(f"Epoch {epoch+1}: Val Accuracy = {acc:.4f}")
        else:
            print("[❌] Data loading or preprocessing failed. Model training aborted.")
            
    except FileNotFoundError as e:
        print(f"[❌] Error during model training: {e}")
    except ValueError as e:
        print(f"[❌] Error during model training: {e}")
    except Exception as e:
        print(f"[❌] An unexpected error occurred during model training: {e}")

    print(f"--- Model Trainer Finished ---")