# main.py
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__)) # This will be aish_test_app/ if main.py is in root
project_root = script_dir # Assuming main.py is in the project root: aish_test_app/
if project_root not in sys.path:
    sys.path.append(project_root)

# Import core modules
from core.log_ingestor import LogIngestorDrain3
from core.aish_runner import main_runner # Updated function name
# You might also want to import and run model_trainer if model doesn't exist
# from core.model_trainer import FinalModelTrainer

def orchestrate_pipeline():
    """
    Orchestrates the AISH-Test offline pipeline:
    1. Ingests and parses raw logs using Drain3.
    2. Runs the AISH-Test simulation (prediction, batch anomaly detection, healing).
    """
    print("🚀 AISH-Test Pipeline Orchestrator Starting 🚀")
    print("-" * 50)

    # --- Configuration ---
    # Paths relative to the project root (aish_test_app/)
    raw_logs_input_path = os.path.join(project_root, "data", "parsed_mozilla_crash_logs.csv")
    processed_logs_output_path = os.path.join(project_root, "data", "processed_logs.csv")
    drain_persistence_file = os.path.join(project_root, "data", "models", "drain3_state.json")
    
    # Model files (check if they exist to suggest training)
    model_file_path = os.path.join(project_root, "data", "models", "lstm_classifier.h5")
    tokenizer_file_path = os.path.join(project_root, "data", "models", "tokenizer.pkl")
    encoder_file_path = os.path.join(project_root, "data", "models", "label_encoder.pkl")

    # --- Step 0: Check for Pre-trained Model (Optional: Add training step) ---
    if not all(os.path.exists(p) for p in [model_file_path, tokenizer_file_path, encoder_file_path]):
        print("[⚠️] Model files not found in data/models/. The AISH runner might fail.")
        print("[INFO] Consider running model_trainer.py if this is the first setup or if retraining is needed.")
        # Example:
        # print("\n[OPTIONAL] Running Model Trainer as model files are missing...")
        # from core.model_trainer import FinalModelTrainer
        # trainer = FinalModelTrainer(
        #     processed_data_path=processed_logs_output_path, # Trainer needs processed logs
        #     models_dir="data/models"
        # )
        # X_data, y_data, _ = trainer.load_and_preprocess_data() # This assumes processed_logs.csv exists
        # trainer.build_and_train_model(X_data, y_data, _)
        # print("[INFO] Model training complete.")
        # print("-" * 50)
    else:
        print("[INFO] Pre-trained model, tokenizer, and encoder found in data/models/.")

    # --- Step 1: Log Ingestion and Parsing with Drain3 ---
    print("\n[TASK 1/2] Parsing Raw Crash Logs with Drain3...")
    try:
        ingestor = LogIngestorDrain3(
            input_csv_path=raw_logs_input_path,
            output_csv_path=processed_logs_output_path,
            persistence_file_path=drain_persistence_file
        )
        ingestor.parse_logs()
        print("[✅] Log ingestion and parsing completed successfully.")
    except FileNotFoundError as e:
        print(f"[❌] Error during log ingestion: {e}")
        print("[INFO] Please ensure 'data/parsed_mozilla_crash_logs.csv' exists.")
        return # Stop if ingestion fails
    except Exception as e:
        print(f"[❌] An unexpected error occurred during log ingestion: {e}")
        return
    print("-" * 50)

    # --- Step 2: Running AISH-Test Core Pipeline (Prediction, Anomaly Detection, Healing) ---
    if os.path.exists(processed_logs_output_path):
        print("\n[TASK 2/2] Running AISH-Test Core Pipeline (Batch Processing)...")
        try:
            # These are the default parameters for main_runner from its own file
            # You can expose them here if needed for easier configuration of main.py
            main_runner( 
                batch_size=50,
                confidence_trigger_threshold=0.6, 
                error_rate_trigger_threshold=0.4
            )
            print("[✅] AISH-Test core pipeline execution completed successfully.")
        except FileNotFoundError as e: # e.g. if model files are missing and Utils fails
            print(f"[❌] Error running AISH-Test core pipeline: {e}")
            print("[INFO] This might be due to missing model files. Ensure model_trainer.py has been run.")
        except Exception as e:
            print(f"[❌] An unexpected error occurred during AISH-Test core pipeline execution: {e}")
    else:
        print(f"[❌] Cannot run AISH-Test core pipeline because '{processed_logs_output_path}' was not created.")
        
    print("-" * 50)
    print("🎉 AISH-Test Pipeline Orchestrator Finished 🎉")

if __name__ == "__main__":
    # Assuming main.py is in the root of your project 'aish_test_app/'
    # and your core modules are in 'aish_test_app/core/'
    # and data is in 'aish_test_app/data/'
    orchestrate_pipeline()