import numpy as np


class DataSplitter:
    """
    A class for splitting datasets into training, validation, and test sets.
    """

    @staticmethod
    def train_test_val_split(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        test_size: float = 0.3,
        val_size: float = 0.0,
        random_seed: int = 42,
    ):
        """
        Splits the dataset into training, validation, and test sets.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            train_size (float): Proportion of the dataset to include in the training set.
            val_size (float): Proportion of the dataset to include in the validation set.
            test_size (float): Proportion of the dataset to include in the test set.
            random_seed (int): Random seed for reproducibility.
        Returns:
            tuple: Tuples of (X_train, y_train), (X_val, y_val), (X_test, y_test).
        """

        assert (
            train_size + val_size + test_size == 1.0
        ), "Train, validation, and test sizes must sum to 1."

        np.random.seed(random_seed)

        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)

        train_end = int(train_size * num_samples)
        val_end = train_end + int(val_size * num_samples)

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
