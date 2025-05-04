# dashboard/streamlit_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

import streamlit as st
import pandas as pd
from core.utils import Utils
from core.anomaly_detector import AnomalyDetector
from core.healer import Healer

st.set_page_config(page_title="AISH-Test Dashboard", layout="wide")

# Construct absolute paths
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "data", "models", "lstm_classifier.h5"))
tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "data", "models", "tokenizer.pkl"))
encoder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "data", "models", "label_encoder.pkl"))


# Load models
utils = Utils(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    encoder_path=encoder_path
)
anomaly_detector = AnomalyDetector()
healer = Healer()

# Sidebar
st.sidebar.title("⚙️ AISH-Test Controls")
uploaded_file = st.sidebar.file_uploader("Upload a Test Log File (.csv with 'cleaned_log')", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    logs = df['cleaned_log'].astype(str).tolist()

    st.sidebar.success(f"Loaded {len(logs)} logs.")

    if st.sidebar.button("🚀 Run AISH-Test"):
        st.title("🧪 AISH-Test Self-Healing System")

        predictions = utils.predict_logs(logs)
        results_df = utils.create_output_dataframe(logs, predictions)

        st.subheader("📋 Log Predictions")
        st.dataframe(results_df)

        st.subheader("🚨 Anomaly Detection and Healing")
        anomalies_detected = []
        healing_actions = []

        for idx, (log, (pred_class, prob)) in enumerate(zip(logs, predictions)):
            is_low_conf, is_high_error = anomaly_detector.evaluate_prediction(prob, pred_class)

            if is_low_conf:
                anomalies_detected.append((idx, log, "Low Confidence"))
                healer.heal(anomaly_type="low_confidence", target="ci-test-service")

            if is_high_error:
                anomalies_detected.append((idx, log, "High Error Rate"))
                healer.heal(anomaly_type="high_error_rate", target="ci-test-service")

        if anomalies_detected:
            st.error(f"⚡ {len(anomalies_detected)} Anomalies Detected!")
            for idx, log, reason in anomalies_detected:
                st.warning(f"[Log {idx}] {reason}: {log[:80]}...")
        else:
            st.success("✅ No anomalies detected.")

        st.subheader("📊 Summary Metrics")
        success_rate = (results_df['Predicted Class'] == 'normal').mean()
        st.metric(label="Success Rate", value=f"{success_rate*100:.2f}%")
