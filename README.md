# AISH-Test: AI Self-Healing Software Testing Pipeline

This application implements the framework from our thesis:
✅ Predicts test failures from logs  
✅ Explains root cause using LangChain + GPT  
✅ Detects anomalies  
✅ Automatically simulates healing/remediation  
✅ Dashboards real-time metrics via Streamlit

## Run the App

1. Install dependencies:
   pip install -r requirements.txt

2. Prepare your log dataset in `data/sample_logs/`

3. Start the main pipeline:
   python main.py

4. (Optional) Launch the dashboard:
   streamlit run dashboard/streamlit_app.py
