# core/anomaly_detector.py

import numpy as np

class AnomalyDetector:
    """
    Anomaly Detector for AISH-Test.
    Flags low-confidence predictions and abnormal error rates.
    """

    def __init__(self, confidence_threshold=0.6, error_rate_window=10, error_rate_threshold=0.4):
        """
        Parameters:
            confidence_threshold (float): minimum probability to trust a prediction
            error_rate_window (int): number of recent predictions to calculate error rate
            error_rate_threshold (float): threshold of abnormal error rate
        """
        self.confidence_threshold = confidence_threshold
        self.error_rate_window = error_rate_window
        self.error_rate_threshold = error_rate_threshold
        self.prediction_history = []

    def check_low_confidence(self, prediction_score):
        """
        Check if a prediction's probability is too low (anomaly).
        """
        if prediction_score < self.confidence_threshold:
            return True
        return False

    def update_prediction_history(self, is_error):
        """
        Maintain sliding window of prediction results (1 = error, 0 = normal).
        """
        self.prediction_history.append(is_error)
        if len(self.prediction_history) > self.error_rate_window:
            self.prediction_history.pop(0)

    def check_error_rate_anomaly(self):
        """
        Check if recent errors exceed expected error rate.
        """
        if len(self.prediction_history) < self.error_rate_window:
            return False  # Not enough data yet

        error_rate = np.mean(self.prediction_history)
        if error_rate >= self.error_rate_threshold:
            return True
        return False

    def evaluate_prediction(self, prediction_score, predicted_class):
        """
        Main interface to evaluate a new prediction:
        - Checks low confidence
        - Updates history
        - Checks rolling error rate
        Returns: (is_low_confidence, is_error_rate_anomaly)
        """
        is_low_conf = self.check_low_confidence(prediction_score)

        is_error = 1 if predicted_class == "failure" else 0
        self.update_prediction_history(is_error)

        is_high_error_rate = self.check_error_rate_anomaly()

        return is_low_conf, is_high_error_rate