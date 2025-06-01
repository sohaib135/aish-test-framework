# core/aish_runner.py
import pandas as pd
import os
import sys
import glob # Keep for potential future use, though not strictly needed now for logs

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.utils import Utils
from core.anomaly_detector import AnomalyDetector
from core.healer import Healer
# from core.explainer_langchain import explain_log # Assuming you'll use this later

def main_runner(batch_size=50, confidence_trigger_threshold=0.6, error_rate_trigger_threshold=0.4):
    print("\n🚀 Starting AISH-Test Real-Time Execution (Batch Mode)")

    # Define paths relative to the project root
    # Input logs (Drain3 parsed templates)
    processed_log_file_path = os.path.join(project_root, "data", "processed_logs.csv")
    
    # Directory for output results
    results_dir = os.path.join(project_root, "data", "results")
    # Full path for the execution results CSV
    execution_results_file_path = os.path.join(results_dir, "aish_execution_summary.csv")

    # --- Initialization ---
    try:
        # Utils will load models from data/models/ by default
        utils_instance = Utils(models_base_dir="data/models") 
        anomaly_detector = AnomalyDetector(
            confidence_threshold=confidence_trigger_threshold, # For individual low confidence
            error_rate_threshold=error_rate_trigger_threshold # For batch error rate
        )
        healer_instance = Healer()
    except FileNotFoundError as e:
        print(f"[❌] Critical Error: Failed to initialize core components: {e}")
        print(f"[INFO] Please ensure models are trained (run model_trainer.py) and all paths are correct.")
        return
    except Exception as e:
        print(f"[❌] Critical Error during initialization: {e}")
        return

    # --- Load Test Logs ---
    if not os.path.exists(processed_log_file_path):
        print(f"[❌] Processed log file not found at '{processed_log_file_path}'.")
        print(f"[INFO] Please run log_ingestor.py to generate 'processed_logs.csv' from your raw data.")
        return

    print(f"[INFO] Loading logs for AISH runner from: {processed_log_file_path}")
    try:
        df_logs = pd.read_csv(processed_log_file_path)
        if 'parsed_log' not in df_logs.columns:
            print(f"[❌] Column 'parsed_log' not found in '{processed_log_file_path}'. Check the CSV content.")
            return
        # Ensure logs are strings and handle potential NaNs by dropping or filling
        df_logs.dropna(subset=['parsed_log'], inplace=True)
        logs_to_process = df_logs['parsed_log'].astype(str).tolist()
    except pd.errors.EmptyDataError:
        print(f"[❌] The log file '{processed_log_file_path}' is empty.")
        return
    except Exception as e:
        print(f"[❌] Error loading or processing log file '{processed_log_file_path}': {e}")
        return

    if not logs_to_process:
        print("[⚠️] No logs found in the input file after cleaning. Exiting.")
        return

    # --- Prediction and Healing Loop ---
    print(f"[📋] Starting prediction on {len(logs_to_process)} log entries...")
    
    try:
        # predict_logs returns list of (predicted_class_str, probability_float)
        all_predictions_with_probs = utils_instance.predict_logs(logs_to_process)
    except Exception as e:
        print(f"[❌] Error during log prediction: {e}")
        return

    if not all_predictions_with_probs:
        print("[❌] Prediction step returned no results. Cannot proceed.")
        return

    execution_summary_log = [] # To store data for the final CSV
    
    # Batch processing variables
    current_batch_predictions = [] # Store (predicted_class, prob) for the current batch
    
    for idx, (log_text, (predicted_class, prob)) in enumerate(zip(logs_to_process, all_predictions_with_probs)):
        log_display_id = idx + 1
        print(f"\n--- [Log {log_display_id}/{len(logs_to_process)}] ---")
        print(f"Text: \"{log_text[:100]}...\"") # Print more of the log
        print(f"  → Predicted: {predicted_class} (Confidence: {prob:.4f})")

        current_batch_predictions.append({'class': predicted_class, 'prob': prob, 'log_text': log_text})
        
        # Store individual log result for the final summary CSV
        execution_summary_log.append({
            "Log ID": log_display_id,
            "Log Text": log_text,
            "Predicted Class": predicted_class,
            "Prediction Confidence": prob,
            "Batch Number": (idx // batch_size) + 1,
            "Anomaly Triggered": "No", # Default, will be updated if healing occurs for its batch
            "Healing Action": "N/A"    # Default
        })

        # Check for batch end or if it's the last log
        is_last_log = (idx + 1) == len(logs_to_process)
        is_batch_boundary = (idx + 1) % batch_size == 0

        if is_batch_boundary or is_last_log:
            batch_number = (idx // batch_size) + 1
            print(f"\n[BATCH {batch_number} PROCESSING] ({len(current_batch_predictions)} logs in this batch)")

            # Analyze current batch
            batch_failure_count = sum(1 for p in current_batch_predictions if p['class'] == 'failure')
            batch_low_confidence_count = sum(1 for p in current_batch_predictions if p['prob'] < anomaly_detector.confidence_threshold)
            
            actual_batch_size = len(current_batch_predictions)
            if actual_batch_size == 0: # Should not happen if logs_to_process is not empty
                print("[WARN] Empty batch encountered. Skipping analysis for this batch.")
                current_batch_predictions = [] # Reset for next batch
                continue

            batch_error_rate = batch_failure_count / actual_batch_size
            batch_low_conf_rate = batch_low_confidence_count / actual_batch_size

            print(f"  Batch {batch_number} Stats: Error Rate: {batch_error_rate:.2f}, Low Confidence Rate: {batch_low_conf_rate:.2f}")

            anomaly_detected_for_batch = False
            healing_anomaly_type = "none"

            if batch_error_rate >= anomaly_detector.error_rate_threshold:
                print(f"  [!] Batch Anomaly: High error rate detected ({batch_error_rate:.2f} >= {anomaly_detector.error_rate_threshold}).")
                anomaly_detected_for_batch = True
                healing_anomaly_type = "high_batch_error_rate" # Specific type for healer
            
            # Optional: Trigger for high low_confidence_rate in batch (can be combined with error rate)
            # For now, using the combined type if either condition is met as per original logic.
            # We can make this more granular if needed.
            # if batch_low_conf_rate >= some_batch_low_conf_threshold:
            #    print(f"  [!] Batch Anomaly: High low-confidence rate detected.")
            #    anomaly_detected_for_batch = True
            #    healing_anomaly_type = "high_batch_low_confidence_rate"


            if anomaly_detected_for_batch:
                print(f"  [🚨] Healing to be triggered for Batch {batch_number} due to: {healing_anomaly_type}")
                
                # Example: Pass info about the batch or a specific problematic log
                target_info_for_healer = {
                    'batch_number': batch_number,
                    'error_rate': batch_error_rate,
                    'job_name': f'ci_batch_job_{batch_number}' # Example target
                }
                # Use the specific anomaly type determined above
                healer_instance.heal(anomaly_type=healing_anomaly_type, target_info=target_info_for_healer)
                
                # Update summary log for items in this batch
                start_index_of_batch = idx - actual_batch_size + 1
                for i in range(start_index_of_batch, idx + 1):
                    execution_summary_log[i]["Anomaly Triggered"] = "Yes"
                    execution_summary_log[i]["Healing Action"] = healing_anomaly_type
            else:
                print(f"  [✅] Batch {batch_number} processed. No batch-level anomalies requiring healing.")

            # Reset for the next batch
            current_batch_predictions = []
            
    # --- Save Full Execution Results ---
    os.makedirs(results_dir, exist_ok=True)
    results_summary_df = pd.DataFrame(execution_summary_log)
    results_summary_df.to_csv(execution_results_file_path, index=False)
    print(f"\n[✅] Full execution summary saved to: {execution_results_file_path}")
    print(f"--- AISH-Test Runner Finished ---")

if __name__ == "__main__":
    # Configuration for the runner
    CONFIG_BATCH_SIZE = 50  # How many logs to process before evaluating batch for healing
    CONFIG_LOW_CONF_THRESHOLD = 0.6 # Individual prediction confidence threshold for 'low confidence'
    CONFIG_BATCH_ERROR_RATE_THRESHOLD = 0.4 # If error rate in a batch exceeds this, trigger healing

    main_runner(
        batch_size=CONFIG_BATCH_SIZE,
        confidence_trigger_threshold=CONFIG_LOW_CONF_THRESHOLD,
        error_rate_trigger_threshold=CONFIG_BATCH_ERROR_RATE_THRESHOLD
    )