# core/aish_runner.py

import pandas as pd
from core.utils import Utils
from core.anomaly_detector import AnomalyDetector
from core.healer import Healer
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main(batch_size=50):
    print("\n🚀 Starting AISH-Test Real-Time Execution (Batch Mode)")

    # Load models
    utils = Utils(
        model_path=os.path.join("data", "models", "lstm_classifier.h5"),
        tokenizer_path=os.path.join("data", "models", "tokenizer.pkl"),
        encoder_path=os.path.join("data", "models", "label_encoder.pkl")
    )
    anomaly_detector = AnomalyDetector()
    healer = Healer()

    # Load test logs
    log_file = os.path.join("data", "processed_logs.csv")
    if not os.path.exists(log_file):
        print("[❌] No processed_logs.csv found. Please run log_ingestor.py first.")
        return

    df = pd.read_csv(log_file)
    logs = df['parsed_log'].astype(str).tolist()

    if not logs:
        print("[❌] No logs found after parsing.")
        return

    # Predict
    print(f"[📋] Predicting on {len(logs)} logs...")
    predictions = utils.predict_logs(logs)

    results_log = []
    batch_errors = 0
    batch_low_confidence = 0

    for idx, (log_text, (predicted_class, prob)) in enumerate(zip(logs, predictions)):
        print(f"\n[Log {idx+1}] {log_text[:80]}...")
        print(f"  → Predicted: {predicted_class} (Confidence: {prob:.2f})")

        is_low_conf, is_high_error = anomaly_detector.evaluate_prediction(prob, predicted_class)

        if is_low_conf:
            batch_low_confidence += 1
        if predicted_class == "failure":
            batch_errors += 1

        results_log.append({
            "Log Text": log_text,
            "Predicted Class": predicted_class,
            "Prediction Confidence": prob
        })

        # At batch end — perform healing if needed
        if (idx + 1) % batch_size == 0 or (idx + 1) == len(logs):
            error_rate = batch_errors / batch_size
            low_conf_rate = batch_low_confidence / batch_size
            print(f"\n[📈] Batch {idx // batch_size + 1} Summary: Error Rate: {error_rate:.2f}, Low Confidence Rate: {low_conf_rate:.2f}")

            if error_rate >= anomaly_detector.error_rate_threshold or low_conf_rate >= anomaly_detector.error_rate_threshold:
                print(f"[🚨] Healing triggered for Batch {idx // batch_size + 1}")
                healer.heal(anomaly_type="high_error_rate", target="ci-batch-service")

            # Reset counters for next batch
            batch_errors = 0
            batch_low_confidence = 0

    # Save results
    results_df = pd.DataFrame(results_log)
    os.makedirs(os.path.join("data", "results"), exist_ok=True)
    results_path = os.path.join("data", "results", "full_execution_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\n[✅] Full execution log saved at: {results_path}")
    print("[🎯] AISH-Test run completed successfully.")

if __name__ == "__main__":
    main()
