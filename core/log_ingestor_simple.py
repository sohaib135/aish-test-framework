# core/log_ingestor_simple.py
import os
import pandas as pd
import re


class SimpleLogIngestor:
    def __init__(self, log_dir, output_csv):
        self.log_dir = log_dir
        self.output_csv = output_csv

    def classify_log(self, line):
        line = line.lower()
        if "error" in line or "critical" in line:
            return "failure"
        elif "timeout" in line or "not found" in line:
            return "failure"
        return "normal"

    def clean_log_line(self, line):
        line = re.sub(r"\[.*?\]", "", line)
        line = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", line)
        line = re.sub(r"\d{2}:\d{2}:\d{2}", "<TIME>", line)
        line = re.sub(r"\d+", "<NUM>", line)
        return line.strip()

    def process_logs(self):
        data = []
        for filename in os.listdir(self.log_dir):
            if not filename.endswith(".log"):
                continue
            with open(os.path.join(self.log_dir, filename), "r") as file:
                for line in file:
                    clean_line = self.clean_log_line(line)
                    label = self.classify_log(line)
                    data.append({
                        "original_log": line.strip(),
                        "cleaned_log": clean_line,
                        "label": label
                    })

        df = pd.DataFrame(data)
        df.to_csv(self.output_csv, index=False)
        print(f"[✅] Saved parsed logs to: {self.output_csv} ({len(df)} rows)")


# Run script directly
if __name__ == "__main__":
    ingestor = SimpleLogIngestor(
        log_dir="C:\\Users\\masad\\Desktop\\Sohaib Research Project\\Project Start\\aish_test_app\\data\\sample_logs",
        output_csv="C:\\Users\\masad\\Desktop\\Sohaib Research Project\\Project Start\\aish_test_app\\data\\processed_logs.csv"
    )
    ingestor.process_logs()