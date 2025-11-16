import numpy as np


class Scaler:
    """
    A class containing various methods for scaling and normalizing numerical data columns
    """

    @staticmethod
    def min_max_normalize_column(col: np.ndarray) -> np.ndarray:
        """
        Performs min-max normalization on a passed column of numerical data.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with normalized values between 0 and 1.
        """

        min_val = np.min(col)
        max_val = np.max(col)
        normalized_col = (col - min_val) / (max_val - min_val)
        return normalized_col

    @staticmethod
    def standardize_column(col: np.ndarray) -> np.ndarray:
        """
        Standardizes a passed column of numerical data to have a mean of 0 and standard deviation of 1.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with standardized values.
        """

        mean = np.mean(col)
        std_dev = np.std(col)
        standardized_col = (col - mean) / std_dev
        return standardized_col

    @staticmethod
    def robust_scale_column(col: np.ndarray) -> np.ndarray:
        """
        Scales a passed column of numerical data using the median and IQR.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with robust scaled values.
        """

        median = np.median(col)
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        robust_scaled_col = (col - median) / iqr
        return robust_scaled_col

    @staticmethod
    def log_transform_column(col: np.ndarray) -> np.ndarray:
        """
        Applies log transformation to a passed column of numerical data.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with log-transformed values.
        """

        log_transformed_col = np.log1p(col)  # log1p is used to handle zero values
        return log_transformed_col

    @staticmethod
    def mean_normalize_column(col: np.ndarray) -> np.ndarray:
        """
        Performs mean normalization on a passed column of numerical data.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with mean-normalized values.
        """

        mean = np.mean(col)
        min_val = np.min(col)
        max_val = np.max(col)
        mean_normalized_col = (col - mean) / (max_val - min_val)
        return mean_normalized_col

    @staticmethod
    def max_abs_scale_column(col: np.ndarray) -> np.ndarray:
        """
        Scales a passed column of numerical data to the range [-1, 1] based on the maximum absolute value.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with max-abs scaled values.
        """

        max_abs_val = np.max(np.abs(col))
        max_abs_scaled_col = col / max_abs_val
        return max_abs_scaled_col

    @staticmethod
    def normalize_columns(
        cols: dict[str, np.ndarray], method: str
    ) -> dict[str, np.ndarray]:
        """
        Normalizes multiple columns using the specified method.
        Args:
            cols (dict[str, np.ndarray]): A dictionary where keys are column names and values are numpy arrays of numerical data.
            method (str): The normalization method to use ('min_max', 'standard', 'robust', 'log', 'mean', 'max_abs').
        Returns:
            dict[str, np.ndarray]: A dictionary with normalized columns.
        """

        for col_name, col_data in cols.items():
            if method == "min_max":
                cols[col_name] = Scaler.min_max_normalize_column(col_data)
            elif method == "standard":
                cols[col_name] = Scaler.standardize_column(col_data)
            elif method == "robust":
                cols[col_name] = Scaler.robust_scale_column(col_data)
            elif method == "log":
                cols[col_name] = Scaler.log_transform_column(col_data)
            elif method == "mean":
                cols[col_name] = Scaler.mean_normalize_column(col_data)
            elif method == "max_abs":
                cols[col_name] = Scaler.max_abs_scale_column(col_data)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
        return cols
