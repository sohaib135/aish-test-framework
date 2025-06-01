# core/log_ingestor.py

import pandas as pd
import os
import sys

# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from drain3 import TemplateMiner
# from drain3.template_miner_config import TemplateMinerConfig # Using default config is fine for Drain3
from drain3.file_persistence import FilePersistence # Corrected import if needed, usually TemplateMiner handles persistence path

class LogIngestorDrain3:
    """
    Real Log Ingestor using Drain3 Parsing.
    Parses 'cleaned_log' from input_csv and saves 'parsed_log' (template) and 'label' to output_csv.
    """

    def __init__(self, input_csv_path, output_csv_path, persistence_file_path="drain3_persistence.json"):
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.persistence_file_path = persistence_file_path

        # Ensure the directory for the persistence file exists
        persistence_dir = os.path.dirname(self.persistence_file_path)
        if persistence_dir and not os.path.exists(persistence_dir):
            os.makedirs(persistence_dir, exist_ok=True)
            print(f"[INFO] Created directory for Drain3 persistence: {persistence_dir}")

        # Using default configuration for TemplateMiner
        # For custom config:
        # from drain3.template_miner_config import TemplateMinerConfig
        # config = TemplateMinerConfig()
        # config.load_from_file("path/to/drain3.ini") # if you have a config file
        # self.template_miner = TemplateMiner(FilePersistence(self.persistence_file_path), config)
        
        self.template_miner = TemplateMiner(FilePersistence(self.persistence_file_path))
        print(f"[INFO] Drain3 TemplateMiner initialized with persistence at: {self.persistence_file_path}")


    def parse_logs(self):
        if not os.path.exists(self.input_csv_path):
            print(f"[❌] Input file {self.input_csv_path} not found!")
            return # Changed from raise to print and return for graceful exit

        print(f"[INFO] Reading input CSV from: {self.input_csv_path}")
        df = pd.read_csv(self.input_csv_path)

        if 'cleaned_log' not in df.columns or 'label' not in df.columns:
            print(f"[❌] Required columns 'cleaned_log' or 'label' not found in {self.input_csv_path}.")
            return

        structured_logs = []
        print(f"[INFO] Starting Drain3 parsing for {len(df)} logs...")
        processed_count = 0
        for idx, row in df.iterrows():
            log_line = str(row['cleaned_log']) # Ensure it's a string
            label = str(row['label'])

            # Add log message to Drain3. It returns a dict with 'template_mined', 'change_type', etc.
            result = self.template_miner.add_log_message(log_line)
            template = result['template_mined'] # The log template

            structured_logs.append({
                "parsed_log": template, # This is the Drain3 template
                "label": label
            })
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"[INFO] Drain3 processed {processed_count}/{len(df)} logs...")
        
        print(f"[INFO] Drain3 parsing completed. Total templates found: {len(self.template_miner.drain.clusters)}")

        if not structured_logs:
            print("[⚠️] No logs were parsed by Drain3. Output CSV will be empty or not created.")
            return

        structured_df = pd.DataFrame(structured_logs)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(self.output_csv_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Created output directory: {output_dir}")

        structured_df.to_csv(self.output_csv_path, index=False)
        print(f"[✅] Parsed logs (templates) saved to {self.output_csv_path}")
        print(f"[INFO] First 5 parsed templates:\n{structured_df.head()}")


if __name__ == "__main__":
    # Define project root and paths relative to it
    # script_dir is core/, project_root is aish_test_app/
    
    # Input: Original cleaned logs
    input_file_path = os.path.join(project_root, "data", "parsed_mozilla_crash_logs.csv")
    
    # Output: Logs parsed into templates by Drain3
    output_file_path = os.path.join(project_root, "data", "processed_logs.csv")

    # Drain3 persistence file (stores learned templates)
    # Place it in the data/models directory for organization
    drain_persistence_path = os.path.join(project_root, "data", "models", "drain3_state.json") 

    print(f"--- Running Log Ingestor (Drain3 Parsing) ---")
    ingestor = LogIngestorDrain3(
        input_csv_path=input_file_path, 
        output_csv_path=output_file_path,
        persistence_file_path=drain_persistence_path
    )
    ingestor.parse_logs()
    print(f"--- Log Ingestor Finished ---")