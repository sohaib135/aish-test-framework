# dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time 

# --- Path Setup ---
# Assuming this script is in a 'dashboard' subdirectory of the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))

# Add project_root and core to sys.path if not already there
# This helps Streamlit find your custom modules when run with `streamlit run`
if project_root not in sys.path:
    sys.path.insert(0, project_root) 
if os.path.join(project_root, 'core') not in sys.path:
    sys.path.insert(0, os.path.join(project_root, 'core'))

# --- Import Core Modules ---
# These imports should now work correctly if sys.path is set up
try:
    from core.utils import Utils
    from core.anomaly_detector import AnomalyDetector
    from core.healer import Healer
    from core.explainer_langchain import LogExplainer # Import the class
except ImportError as e:
    st.error(f"Failed to import core modules: {e}. Ensure 'core' directory is in Python path and files exist.")
    st.stop() # Stop execution if core modules can't be loaded

# --- Page Configuration ---
st.set_page_config(
    page_title="AISH-Test Dashboard",
    page_icon="🧪", # Can be an emoji or path to an .ico file
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Core Components (Models, Tokenizer, Encoder, Explainer) ---
@st.cache_resource
def load_core_components():
    try:
        utils_instance = Utils(models_base_dir="data/models") # Relative to project_root
        anomaly_detector_instance = AnomalyDetector(confidence_threshold=0.6, error_rate_threshold=0.4)
        healer_instance = Healer()
        log_explainer_instance = LogExplainer() # Initialize your explainer
        return utils_instance, anomaly_detector_instance, healer_instance, log_explainer_instance
    except FileNotFoundError as e:
        st.error(f"🔴 Critical Error: Failed to load model/tokenizer/encoder: {e}")
        st.info("Please ensure 'model_trainer.py' has been run successfully and model files are in 'data/models'.")
        return None, None, None, None
    except Exception as e:
        st.error(f"🔴 An unexpected error occurred while loading core components: {e}")
        return None, None, None, None

utils, anomaly_detector, healer, log_explainer = load_core_components()

# --- Initialize Session State ---
# (Using a function for cleaner initialization)
def init_session_state():
    defaults = {
        'run_analysis': False,
        'results_df': pd.DataFrame(),
        'batch_analysis_summary': [],
        'overall_stats': {},
        'uploaded_filename': None,
        'raw_logs_for_display': [],
        'active_tab': "📋 Individual Log Predictions"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- UI Styling ---
st.markdown("""
    <style>
        /* Add custom styles here if needed */
        .stButton>button {
            font-weight: bold;
        }
        .stMetric {
            background-color: #FFFFFF; /* Light background for metric cards */
            border: 1px solid #E0E0E0; /* Light border */
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Subtle shadow */
        }
        .block-container { /* Main content area padding */
            padding-top: 2rem; 
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    # Attempt to load logo, handle FileNotFoundError gracefully
    logo_path = os.path.join(project_root, "docs", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.info("Logo not found at docs/logo.png. You can add one there.")
        
    st.title("🧪 AISH-Test Controls")
    st.markdown("AI-Powered Self-Healing for CI/CD Test Logs")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Test Log File (.csv)",
        type="csv",
        help="CSV must contain a column named 'cleaned_log' with raw log messages."
    )

    if uploaded_file:
        if st.session_state.uploaded_filename != uploaded_file.name:
            # Reset state for a new file upload
            init_session_state() # Reset all relevant keys
            st.session_state.uploaded_filename = uploaded_file.name
            st.success(f"File '{uploaded_file.name}' loaded. Ready for analysis.")
        
        st.subheader("⚙️ Analysis Configuration")
        batch_size_config = st.slider("Batch Size for Analysis", min_value=10, max_value=200, value=50, step=10, key="batch_size_slider")
        
        # Ensure anomaly_detector is loaded before accessing its attributes
        if anomaly_detector:
            current_error_threshold = anomaly_detector.error_rate_threshold
            error_rate_threshold_config = st.slider("Batch Error Rate Healing Threshold", 
                                                    min_value=0.1, max_value=1.0, 
                                                    value=float(current_error_threshold), # Use current value
                                                    step=0.05, key="error_rate_slider")
            if error_rate_threshold_config != current_error_threshold:
                anomaly_detector.error_rate_threshold = error_rate_threshold_config
                st.rerun() # Rerun if threshold changes to reflect in subsequent analysis
        else:
            st.slider("Batch Error Rate Healing Threshold", 0.1, 1.0, 0.4, 0.05, key="error_rate_slider_disabled", disabled=True)


        if st.button("🚀 Analyze & Heal Logs", type="primary", use_container_width=True, key="analyze_button"):
            if not all([utils, anomaly_detector, healer, log_explainer]):
                st.error("Core components not loaded. Cannot run analysis. Check console for errors.")
            else:
                st.session_state.run_analysis = True
                # Clear previous results for a fresh run
                st.session_state.results_df = pd.DataFrame()
                st.session_state.batch_analysis_summary = []
                st.session_state.overall_stats = {}
                st.session_state.raw_logs_for_display = []


                with st.spinner("🔬 Processing logs, predicting failures, and simulating healing actions..."):
                    try:
                        df_input = pd.read_csv(uploaded_file)
                        if 'cleaned_log' not in df_input.columns:
                            st.error("Uploaded CSV must contain a 'cleaned_log' column.")
                            st.session_state.run_analysis = False
                        else:
                            raw_logs = df_input['cleaned_log'].astype(str).dropna().tolist()
                            st.session_state.raw_logs_for_display = raw_logs # Store for display

                            if not raw_logs:
                                st.warning("No valid logs found in 'cleaned_log' column after cleaning.")
                                st.session_state.run_analysis = False
                            else:
                                predictions_with_probs = utils.predict_logs(raw_logs)
                                st.session_state.results_df = utils.create_output_dataframe(raw_logs, predictions_with_probs)
                                
                                temp_batch_summary = []
                                total_logs_count = len(raw_logs)
                                total_predicted_failures = 0
                                total_low_confidence_preds = 0
                                healing_events_triggered = 0

                                for i in range(0, total_logs_count, batch_size_config):
                                    batch_raw_logs = raw_logs[i:i + batch_size_config]
                                    batch_preds_probs = predictions_with_probs[i:i + batch_size_config]
                                    
                                    if not batch_preds_probs: continue

                                    batch_num = (i // batch_size_config) + 1
                                    b_failures = sum(1 for _, (p_cls, _) in enumerate(batch_preds_probs) if p_cls == 'failure')
                                    b_low_conf = sum(1 for _, (_, p_prb) in enumerate(batch_preds_probs) if p_prb < anomaly_detector.confidence_threshold)
                                    
                                    total_predicted_failures += b_failures
                                    total_low_confidence_preds += b_low_conf

                                    actual_size = len(batch_preds_probs)
                                    b_err_rate = b_failures / actual_size if actual_size > 0 else 0
                                    
                                    batch_item = {
                                        "Batch #": batch_num, "Logs": actual_size, "Failures": b_failures,
                                        "Error Rate": f"{b_err_rate:.2%}", "Low Confidence": b_low_conf,
                                        "Healing Triggered": "No", "Healing Type": "N/A"
                                    }

                                    if b_err_rate >= anomaly_detector.error_rate_threshold:
                                        batch_item["Healing Triggered"] = "Yes"
                                        batch_item["Healing Type"] = "high_batch_error_rate"
                                        healing_events_triggered += 1
                                        # Healer prints to console; for UI, we just log the event
                                        # healer.heal(anomaly_type="high_batch_error_rate", target_info={'batch_number': batch_num})
                                    
                                    temp_batch_summary.append(batch_item)
                                
                                st.session_state.batch_analysis_summary = temp_batch_summary
                                st.session_state.overall_stats = {
                                    "Total Logs Processed": total_logs_count,
                                    "Total Predicted Failures": total_predicted_failures,
                                    "Overall Predicted Failure Rate": f"{(total_predicted_failures / total_logs_count if total_logs_count > 0 else 0):.2%}",
                                    "Total Healing Events Simulated": healing_events_triggered
                                }
                                st.success("✅ Analysis complete! View results in the main panel.")
                                st.session_state.active_tab = "📋 Individual Log Predictions" # Switch to results tab

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)[:500]}...") # Show limited error msg
                        st.session_state.run_analysis = False
    else:
        st.info("Upload a CSV log file to begin.")

    st.divider()
    if st.button("🔄 Reset Dashboard", use_container_width=True, key="reset_button"):
        init_session_state() # Reset all relevant keys
        st.rerun()

# --- Main Page ---
st.title("🔬 AISH-Test: AI Self-Healing Dashboard")
st.markdown("Interactive dashboard for analyzing CI/CD log files, predicting failures, and simulating self-healing actions.")

if not all([utils, anomaly_detector, healer, log_explainer]):
    st.warning("🔴 Dashboard cannot operate as core components failed to load. Please check the console for errors and ensure model files exist in 'data/models/'.")
    st.stop()

if st.session_state.run_analysis and not st.session_state.results_df.empty:
    st.success(f"Displaying analysis for: **{st.session_state.uploaded_filename}**")
    st.divider()

    # --- Overall Statistics Section ---
    st.subheader("📊 Overall Analysis Summary")
    if st.session_state.overall_stats:
        cols_stats = st.columns(3)
        cols_stats[0].metric("Total Logs Processed", st.session_state.overall_stats.get("Total Logs Processed", 0))
        cols_stats[1].metric("Predicted Failure Rate", st.session_state.overall_stats.get("Overall Predicted Failure Rate", "0.00%"))
        cols_stats[2].metric("Healing Events Simulated", st.session_state.overall_stats.get("Total Healing Events Simulated", 0))
    st.divider()

    # --- Tabs for Detailed Results ---
    # Use st.session_state to manage the active tab if needed, or let Streamlit handle it.
    tab_titles = ["📋 Individual Log Predictions", "🔍 Batch Analysis & Healing", "💡 Failure Explanations"]
    tab1, tab2, tab3 = st.tabs(tab_titles)

    with tab1:
        st.header("Individual Log Prediction Details")
        st.markdown(f"Predictions based on the trained LSTM model. Logs are preprocessed using the learned tokenizer.")
        
        def highlight_failures(row):
            color = '#FFCDD2' if row['Predicted Class'] == 'failure' else '' # Light red for failure
            return [f'background-color: {color}'] * len(row)

        st.dataframe(
            st.session_state.results_df.style.apply(highlight_failures, axis=1), 
            height=500, 
            use_container_width=True,
            column_config={
                "Log": st.column_config.TextColumn("Log Message (Raw Input)", width="large"),
                "Predicted Class": st.column_config.TextColumn("Prediction"),
                "Prediction Confidence": st.column_config.NumberColumn("Confidence", format="%.4f")
            }
        )

    with tab2:
        st.header("Batch-wise Anomaly Detection & Healing Simulation")
        if st.session_state.batch_analysis_summary:
            df_batch_summary = pd.DataFrame(st.session_state.batch_analysis_summary)
            st.dataframe(df_batch_summary, use_container_width=True, hide_index=True)
            st.markdown(f"Healing is simulated if a batch's error rate ≥ **{anomaly_detector.error_rate_threshold:.0%}**.")
            st.info("Note: Actual healing actions (like service restarts) are logged to the console by the `Healer` module and simulated here for UI demonstration.")
        else:
            st.info("No batch analysis data available for the current run.")

    with tab3:
        st.header("💡 Explainable AI: Sample Failure Diagnoses")
        st.markdown("Illustrative explanations for some of the predicted failures using the `LogExplainer` module. In a full system, this would leverage LangChain and a Large Language Model.")
        
        failure_logs_df = st.session_state.results_df[st.session_state.results_df['Predicted Class'] == 'failure']
        if not failure_logs_df.empty:
            # Display explanations for a sample of predicted failures
            num_explanations_to_show = min(5, len(failure_logs_df))
            st.info(f"Showing explanations for up to {num_explanations_to_show} sample predicted failures:")

            # Iterate through the original raw logs that were predicted as failures
            original_failure_logs_texts = [
                st.session_state.raw_logs_for_display[i] 
                for i, row in st.session_state.results_df.iterrows() 
                if row['Predicted Class'] == 'failure'
            ]
            
            sampled_failure_texts = np.random.choice(original_failure_logs_texts, size=num_explanations_to_show, replace=False)

            for i, log_text_for_explanation in enumerate(sampled_failure_texts):
                with st.expander(f"🚨 Explanation for Failure Log Sample #{i+1}: \"{log_text_for_explanation[:70]}...\""):
                    explanation_dict = log_explainer.get_explanation(log_text_for_explanation) # Use the class instance
                    st.markdown(f"**Original Log:** `{explanation_dict.get('original_log', 'N/A')}`")
                    st.markdown(f"**🗣️ Explanation:** {explanation_dict.get('explanation', 'No explanation available.')}")
                    
                    causes = explanation_dict.get('potential_causes', [])
                    if causes:
                        st.markdown("**Possible Causes:**")
                        for cause in causes:
                            st.markdown(f"- {cause}")
                            
                    suggestions = explanation_dict.get('suggested_actions', [])
                    if suggestions:
                        st.markdown("**🔧 Suggested Actions:**")
                        for suggestion in suggestions:
                            st.markdown(f"- {suggestion}")
                    if explanation_dict.get("simulated"):
                        st.caption("Note: This explanation is currently rule-based/simulated.")
        else:
            st.success("✅ No failures predicted in the uploaded logs to explain.")
            
    # Display Confusion Matrix (if generated by predictor.py and path is known)
    cm_path = os.path.join(project_root, "data", "results", "evaluation_confusion_matrix.png")
    if os.path.exists(cm_path):
        st.divider()
        with st.expander("📉 View Offline Model Evaluation (Confusion Matrix)", expanded=False):
            st.image(cm_path, caption="Confusion Matrix from offline model evaluation via predictor.py.")

else:
    st.info("✨ Welcome to the AISH-Test Dashboard! Upload a CSV log file containing a `cleaned_log` column using the sidebar to start the analysis.")
    st.markdown("""
        **How to Use:**
        1.  Ensure your AI model (`lstm_classifier.h5`), tokenizer (`tokenizer.pkl`), and label encoder (`label_encoder.pkl`) are trained and located in the `data/models/` directory.
        2.  Prepare a CSV file. It **must** contain a column named `cleaned_log` with your raw (unparsed by Drain3 for this dashboard) log messages.
        3.  Upload the CSV file using the **"Upload Test Log File"** button in the sidebar.
        4.  Adjust batch analysis settings if desired.
        5.  Click **"🚀 Analyze & Heal Logs"** to process the logs.
        6.  View results, batch summaries, and sample failure explanations in the respective tabs.
        
        **Important Notes:**
        - This dashboard uses the pre-trained model to predict on the `cleaned_log` data you upload. It applies the learned tokenizer on-the-fly.
        - 'Healing' actions are simulated for this dashboard (actual healing logs to console).
        - Explanations are currently rule-based simulations of a LangChain/LLM interaction.
    """)

st.divider()
st.caption("AISH-Test Framework - Thesis Project | Sohaib Ahmed")