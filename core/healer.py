# core/healer.py

import time

class Healer:
    """
    A simple self-healing engine for AISH-Test.
    Based on the detected failure or anomaly type, perform appropriate healing actions.
    """

    def __init__(self):
        pass

    def restart_service(self, service_name="default-service"):
        print(f"[🔄] Attempting to restart service: {service_name}")
        time.sleep(1)  # simulate delay
        print(f"[✅] Service {service_name} restarted successfully.")

    def clear_temp_files(self, temp_dir="/tmp"):
        print(f"[🧹] Clearing temporary files in directory: {temp_dir}")
        time.sleep(1)
        print(f"[✅] Temporary files cleared from {temp_dir}.")

    def retry_task(self, task_name="test-job"):
        print(f"[🔁] Retrying task: {task_name}")
        time.sleep(1)
        print(f"[✅] Task {task_name} retried successfully.")

    def send_alert(self, message="Anomaly detected in CI/CD pipeline."):
        print(f"[📢] Sending alert: {message}")
        time.sleep(0.5)
        print(f"[✅] Alert sent to DevOps Team.")

    def heal(self, anomaly_type="low_confidence", target="default-service"):
        """
        Main interface to trigger healing actions based on anomaly type.
        """
        print(f"\n[🛠️] Healing process initiated for anomaly type: {anomaly_type}")

        if anomaly_type == "low_confidence":
            self.restart_service(service_name=target)
        elif anomaly_type == "high_error_rate":
            self.clear_temp_files(temp_dir="/tmp/aish-logs")
            self.retry_task(task_name="build-test-job")
        elif anomaly_type == "hard_failure":
            self.restart_service(service_name=target)
            self.clear_temp_files(temp_dir="/tmp")
            self.retry_task(task_name="critical-task")
            self.send_alert(message="Critical failure detected and actions initiated.")
        else:
            print("[ℹ️] Unknown anomaly type. Manual intervention may be needed.")

        print("[✅] Healing actions completed.\n")
