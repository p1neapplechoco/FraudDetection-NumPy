import numpy as np


class OutlierHandler:
    @staticmethod
    def iqr_based_outlier_bounds(col: np.ndarray) -> tuple:
        """
        Calculates the lower and upper bounds for outlier detection using the IQR method.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            tuple: A tuple containing the lower and upper bounds for outlier detection.
        """
        Q1 = np.percentile(col, 25)
        Q3 = np.percentile(col, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    @staticmethod
    def get_cols_with_outliers(cols: dict, threshold_percentage: float = 10) -> list:
        """
        Identifies columns that contain outliers based on the IQR method.
        Args:
            cols (dict): A dictionary where keys are column names and values are numpy arrays of data.
            threshold_percentage (float): The percentage threshold to consider a column as having outliers.
        Returns:
            list: A list of column names that contain outliers.
        """
        cols_with_outliers = []

        for key in cols.keys():
            col = cols[key]
            lower_bound, upper_bound = OutlierHandler.iqr_based_outlier_bounds(col)
            outliers = col[(col < lower_bound) | (col > upper_bound)]

            if len(outliers) / len(col) * 100 > threshold_percentage:
                cols_with_outliers.append(key)

        return cols_with_outliers

    @staticmethod
    def clip_outliers(
        col: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0
    ) -> np.ndarray:
        """
        Clips outliers in a numerical column based on specified percentiles.
        Args:
            col (np.ndarray): A numpy array of numerical data.
            lower_percentile (float): The lower percentile threshold.
            upper_percentile (float): The upper percentile threshold.
        Returns:
            np.ndarray: A numpy array with outliers clipped.
        """
        lower_bound = np.percentile(col, lower_percentile)
        upper_bound = np.percentile(col, upper_percentile)
        clipped_col = np.clip(col, lower_bound, upper_bound)

        return clipped_col

    @staticmethod
    def clip_outliers_by_iqr(col: np.ndarray) -> np.ndarray:
        """
        Clips outliers in a numerical column using the IQR method.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            np.ndarray: A numpy array with outliers clipped.
        """
        lower_bound, upper_bound = OutlierHandler.iqr_based_outlier_bounds(col)
        clipped_col = np.clip(col, lower_bound, upper_bound)
        return clipped_col

    @staticmethod
    def count_outliers(col: np.ndarray) -> int:
        """
        Counts the number of outliers in a numerical column using the IQR method.
        Args:
            col (np.ndarray): A numpy array of numerical data.
        Returns:
            int: The number of outliers in the column.
        """
        lower_bound, upper_bound = OutlierHandler.iqr_based_outlier_bounds(col)
        outliers = col[(col < lower_bound) | (col > upper_bound)]
        return len(outliers)
