import numpy as np
from typing import Tuple
from pathlib import Path


class DataSplitter:
    """
    A class for splitting datasets into training, validation, and test sets.
    """

    @staticmethod
    def shuffle_data(
        X: np.ndarray, y: np.ndarray, random_seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shuffles the dataset.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            random_seed (int): Random seed for reproducibility.
        Returns:
            np.ndarray: Shuffled feature matrix.
            np.ndarray: Shuffled target vector.
        """
        np.random.seed(random_seed)
        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)
        return X[indices], y[indices]

    @staticmethod
    def train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        test_size: float = 0.3,
        random_seed: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Splits the dataset into training and test sets.
        Args:
            X (np.ndarray): Feature matrix (input features).
            y (np.ndarray): Target vector (labels or desired outputs).
            train_size (float): Proportion of the dataset to include in the training set.
            test_size (float): Proportion of the dataset to include in the test set.
            random_seed (int): Random seed for reproducibility.
        Returns:
            tuple: Tuples of (X_train, y_train) and (X_test, y_test).
        """

        assert train_size + test_size == 1.0, "Train and test sizes must sum to 1."

        np.random.seed(random_seed)

        num_samples = X.shape[0]
        indices = np.random.permutation(num_samples)

        train_end = int(train_size * num_samples)

        train_indices = indices[:train_end]
        test_indices = indices[train_end:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        return (X_train, y_train), (X_test, y_test)

    @staticmethod
    def train_test_val_split(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float = 0.7,
        test_size: float = 0.3,
        val_size: float = 0.0,
        random_seed: int = 42,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Splits the dataset into training, validation, and test sets.
        Args:
            X (np.ndarray): Feature matrix (input features).
            y (np.ndarray): Target vector (labels or desired outputs).

            train_size (float): Proportion of the dataset to include in the training set.
            val_size (float): Proportion of the dataset to include in the validation set.
            test_size (float): Proportion of the dataset to include in the test set.
            random_seed (int): Random seed for reproducibility.
        Returns:
            tuple: Tuples of (X_train, y_train), (X_test, y_test), (X_val, y_val).
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

        return (X_train, y_train), (X_test, y_test), (X_val, y_val)

    @staticmethod
    def save_split_data(
        X: np.ndarray,
        y: np.ndarray,
        train_size: float,
        test_size: float,
        output_dir: Path,
        random_seed: int = 42,
    ) -> None:
        """
        Splits the dataset and saves the training and test sets to disk.
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            train_size (float): Proportion of the dataset to include in the training set.
            test_size (float): Proportion of the dataset to include in the test set.
            output_dir (Path): Directory to save the split datasets.
            random_seed (int): Random seed for reproducibility.
        """
        (X_train, y_train), (X_test, y_test) = DataSplitter.train_test_split(
            X, y, train_size, test_size, random_seed
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_test.npy", y_test)

        print(f"Data saved to {output_dir}")
