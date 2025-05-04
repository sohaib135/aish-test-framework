# main.py

from core.log_ingestor import LogIngestorDrain3
from core.aish_runner import main as run_aish

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def orchestrate():
    print("🚀 [1/2] Parsing crash logs with Drain3...")
    input_path = os.path.join("data", "parsed_mozilla_crash_logs.csv")
    output_path = os.path.join("data", "processed_logs.csv")

    ingestor = LogIngestorDrain3(input_csv=input_path, output_csv=output_path)
    ingestor.parse_logs()

    print("\n🚀 [2/2] Running AISH-Test pipeline...")
    run_aish()

if __name__ == "__main__":
    orchestrate()
