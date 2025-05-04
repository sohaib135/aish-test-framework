# core/log_ingestor.py

import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence

class LogIngestorDrain3:
    """
    Real Log Ingestor using Drain3 Parsing
    """

    def __init__(self, input_csv, output_csv, persistence_file="drain3_persistence.json"):
        self.input_csv = input_csv
        self.output_csv = output_csv

        config = TemplateMinerConfig()
        #config.load_default()
        persistence = FilePersistence(persistence_file)
        self.template_miner = TemplateMiner(persistence, config)

    def parse_logs(self):
        if not os.path.exists(self.input_csv):
            raise FileNotFoundError(f"Input file {self.input_csv} not found!")

        df = pd.read_csv(self.input_csv)

        # Initialize structured logs list
        structured_logs = []

        for idx, row in df.iterrows():
            log_line = row['cleaned_log']
            label = row['label']

            result = self.template_miner.add_log_message(log_line)
            template = result['template_mined'] if result else log_line

            structured_logs.append({
                "parsed_log": template,
                "label": label
            })

        structured_df = pd.DataFrame(structured_logs)
        structured_df.to_csv(self.output_csv, index=False)

        print(f"[✅] Parsed logs saved to {self.output_csv}")

if __name__ == "__main__":
    input_path = os.path.join("data", "parsed_mozilla_crash_logs.csv")
    output_path = os.path.join("data", "processed_logs.csv")

    ingestor = LogIngestorDrain3(input_csv=input_path, output_csv=output_path)
    ingestor.parse_logs()
