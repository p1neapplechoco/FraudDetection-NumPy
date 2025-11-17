import numpy as np
from typing import List, Dict, Optional
import matplotlib.pyplot as plt


class Evaluator:
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the Evaluator class with specified metrics.
        Args:
            metrics (List[str], optional): List of metrics to evaluate. Defaults to None, which includes all metrics.
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score"]

        self.metrics = {
            metric: getattr(self, metric) for metric in metrics if hasattr(self, metric)
        }

    def evaluate(
        self, y_true: np.ndarray, y_pred: np.ndarray, visualize: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the predictions using the specified metrics.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            visualize (bool, optional): Whether to visualize the results. Defaults to False.
        Returns:
            Dict[str, float]: Dictionary containing metric names and their corresponding scores.
        """

        results = {}
        for metric_name, metric_func in self.metrics.items():
            score = metric_func(y_true=y_true, y_pred=y_pred)
            results[metric_name] = score
            print(f"{metric_name}: {score:.4f}")

        if visualize:
            self.__visualize_results(y_true, y_pred)

        return results

    def __visualize_results(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Visualize the evaluation results using a bar chart.
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
        """
        scores = [self.metrics[metric](y_true, y_pred) for metric in self.metrics]
        plt.bar(list(self.metrics.keys()), scores, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title("Evaluation Metrics")
        plt.show()

    @staticmethod
    def accuracy(y_true, y_pred):
        correct_predictions = np.sum(y_true == y_pred)
        total_predictions = len(y_true)
        return correct_predictions / total_predictions

    @staticmethod
    def precision(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        predicted_positives = np.sum(y_pred == 1)
        if predicted_positives == 0:
            return 0.0
        return true_positives / predicted_positives

    @staticmethod
    def recall(y_true, y_pred):
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        actual_positives = np.sum(y_true == 1)
        if actual_positives == 0:
            return 0.0
        return true_positives / actual_positives

    @staticmethod
    def f1_score(y_true, y_pred):
        precision = Evaluator.precision(y_true, y_pred)
        recall = Evaluator.recall(y_true, y_pred)
        if (precision + recall) == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def pr_auc(y_true, y_pred):
        from sklearn.metrics import precision_recall_curve, auc

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)
