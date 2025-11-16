import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


class Resampler:
    """
    A class for handling over-sampling and under-sampling of datasets to handle class imbalance.
    """

    @staticmethod
    def over_sample(
        X: np.ndarray, y: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Over-samples the minority class to reach the target size.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            target_size (int): Desired number of samples in the minority class after resampling.
        Returns:
            np.ndarray: Resampled feature matrix.
            np.ndarray: Resampled target vector.
        """
        counter = Counter(y)
        classes = list(counter.keys())
        class_counts = list(counter.values())

        if len(set(class_counts)) == 1:
            return (X, y)

        minority_class = classes[np.argmin(class_counts)]
        minority_indices = np.where(y == minority_class)[0]

        additional_samples_needed = target_size - counter[minority_class]

        if additional_samples_needed <= 0:
            return (X, y)

        sampled_indices = np.random.choice(
            minority_indices, size=additional_samples_needed, replace=True
        )
        X_resampled = np.vstack((X, X[sampled_indices]))
        y_resampled = np.hstack((y, y[sampled_indices]))

        return (X_resampled, y_resampled)

    @staticmethod
    def under_sample(
        X: np.ndarray, y: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Under-samples the majority class to reach the target size.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            target_size (int): Desired number of samples in the majority class after resampling.
        Returns:
            np.ndarray: Resampled feature matrix.
            np.ndarray: Resampled target vector.
        """
        counter = Counter(y)
        classes = list(counter.keys())
        class_counts = list(counter.values())

        if len(set(class_counts)) == 1:
            return (X, y)

        majority_class = classes[np.argmax(class_counts)]
        majority_indices = np.where(y == majority_class)[0]

        if counter[majority_class] <= target_size:
            return (X, y)

        sampled_indices = np.random.choice(
            majority_indices, size=target_size, replace=False
        )
        minority_indices = np.where(y != majority_class)[0]

        X_resampled = np.vstack((X[sampled_indices], X[minority_indices]))
        y_resampled = np.hstack((y[sampled_indices], y[minority_indices]))

        return (X_resampled, y_resampled)
