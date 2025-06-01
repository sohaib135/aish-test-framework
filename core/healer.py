# core/healer.py

import time
import os # Added for path joining if needed for more complex healing

# It's good practice to have project_root defined if healer needs to access project files
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(script_dir, '..'))

class Healer:
    """
    A simple self-healing engine for AISH-Test.
    Based on the detected failure or anomaly type, perform appropriate healing actions.
    """

    def __init__(self):
        print("[INFO] Healer initialized.")
        # Example: Define a base directory for logs if healer needs to interact with them
        # self.log_output_dir = os.path.join(project_root, "data", "healer_logs")
        # os.makedirs(self.log_output_dir, exist_ok=True)

    def restart_service(self, service_name="default-ci-service"): # Made service name more specific
        print(f"[🔄] Healer: Attempting to restart service: {service_name}")
        time.sleep(0.5)  # simulate delay
        print(f"[✅] Healer: Service {service_name} restarted successfully.")
        # In a real scenario, this would involve shell commands or API calls:
        # e.g., os.system(f"systemctl restart {service_name}")

    def clear_temp_files(self, temp_dir_path="/tmp/aish-ci-cache"): # Made path more specific
        print(f"[🧹] Healer: Clearing temporary files in directory: {temp_dir_path}")
        # Example: os.makedirs(temp_dir_path, exist_ok=True)
        # Example: for f in os.listdir(temp_dir_path): os.remove(os.path.join(temp_dir_path, f))
        time.sleep(0.5) # simulate delay
        print(f"[✅] Healer: Temporary files cleared from {temp_dir_path}.")

    def retry_task(self, task_name="ci-build-test-job"): # Made task name more specific
        print(f"[🔁] Healer: Retrying task: {task_name}")
        time.sleep(0.5) # simulate delay
        print(f"[✅] Healer: Task {task_name} retried successfully.")
        # In a real scenario, this might involve calling a Jenkins API, etc.

    def send_alert(self, message="Anomaly detected in CI/CD pipeline. Healing actions performed."):
        print(f"[📢] Healer: Sending alert: {message}")
        time.sleep(0.2) # simulate delay
        print(f"[✅] Healer: Alert sent to DevOps Team / Monitoring System.")

    def heal(self, anomaly_type="unknown_anomaly", target_info=None): # target_info can be a dict
        """
        Main interface to trigger healing actions based on anomaly type.

        Args:
            anomaly_type (str): The type of anomaly detected.
            target_info (dict, optional): Additional information about the target for healing,
                                         e.g., {'service_name': 'auth-service', 'job_id': 'build-123'}
        """
        print(f"\n[🛠️] Healer: Healing process initiated for ANOMALY TYPE: '{anomaly_type}'")
        if target_info:
            print(f"[INFO] Healer: Target information: {target_info}")

        healed = False
        if anomaly_type == "low_confidence_prediction": # More specific
            service_to_restart = target_info.get('service_name', 'default-ci-service') if target_info else 'default-ci-service'
            self.restart_service(service_name=service_to_restart)
            healed = True
        
        # Handling the combined case from aish_runner.py
        elif anomaly_type == "high_error_rate_or_low_confidence" or anomaly_type == "high_batch_error_rate":
            print("[INFO] Healer: Detected high batch error rate or persistent low confidence.")
            # Define a standard set of actions for this common scenario
            self.clear_temp_files(temp_dir_path="/tmp/aish-ci-cache/batch_errors")
            self.retry_task(task_name=target_info.get('job_name', 'ci-batch-processing-job') if target_info else 'ci-batch-processing-job')
            self.send_alert(f"High error/low confidence batch detected: {anomaly_type}. Retried relevant job.")
            healed = True
            
        elif anomaly_type == "critical_system_failure": # More specific
            service_to_restart = target_info.get('service_name', 'critical-ci-service') if target_info else 'critical-ci-service'
            task_to_retry = target_info.get('task_name', 'critical-validation-task') if target_info else 'critical-validation-task'
            
            self.restart_service(service_name=service_to_restart)
            self.clear_temp_files(temp_dir_path="/tmp/aish-critical-failure")
            self.retry_task(task_name=task_to_retry)
            self.send_alert(message=f"Critical failure ({anomaly_type}) detected. Actions initiated for {service_to_restart}.")
            healed = True
        
        # Add more specific anomaly types if your system can detect them
        # elif anomaly_type == "specific_error_pattern_X":
        #     # Custom healing logic for pattern X
        #     print("[INFO] Healer: Handling specific error pattern X...")
        #     healed = True

        else:
            print(f"[⚠️] Healer: Unknown or unhandled ANOMALY TYPE: '{anomaly_type}'. Manual intervention may be required.")
            self.send_alert(message=f"Unknown anomaly type '{anomaly_type}' detected. Please investigate.")
            healed = False # Explicitly state not healed by this logic branch

        if healed:
            print("[✅] Healer: Healing actions completed successfully.")
        else:
            print("[ℹ️] Healer: Healing process concluded for this anomaly type (may require manual check).")
        print("-" * 30)

if __name__ == '__main__':
    print("--- Testing Healer ---")
    healer_instance = Healer()

    print("\nSimulating low confidence scenario:")
    healer_instance.heal(anomaly_type="low_confidence_prediction", target_info={'service_name': 'user-auth-service'})
    
    print("\nSimulating high error rate batch scenario:")
    healer_instance.heal(anomaly_type="high_error_rate_or_low_confidence", target_info={'job_name': 'data-processing-batch-job'})

    print("\nSimulating critical failure:")
    healer_instance.heal(anomaly_type="critical_system_failure", target_info={'service_name': 'payment-gateway', 'task_name': 'transaction-processor'})
    
    print("\nSimulating unknown anomaly:")
    healer_instance.heal(anomaly_type="undefined_glitch_type_XYZ")
    
    print("--- Healer Test Finished ---")
