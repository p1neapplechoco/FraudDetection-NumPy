import numpy as np


class Cleaner:
    """
    A class containing static methods for basic data cleaning tasks.
    """

    @staticmethod
    def convert_cols_to_numeric(cols: dict) -> dict:
        """
        Converts the values in the columns of a dataset to numeric (float) type.
        Args:
            cols (dict): A dictionary where keys are column names and values are lists of data.
        Returns:
            dict: A dictionary with the same keys but with values converted to numpy arrays of floats.
        """
        for key in cols.keys():
            try:
                cols[key] = np.array([float(value) for value in cols[key]])
            except (ValueError, SyntaxError):
                cols[key] = np.array(
                    [float(value.replace('"', "")) for value in cols[key]]
                )

        return cols

    @staticmethod
    def clean_string(s: str) -> str:
        """
        Cleans a string by removing leading/trailing whitespace and converting to lowercase.
        Args:
            s (str): The input string.
        Returns:
            str: The cleaned string.
        """
        return s.lower().strip().replace('"', "").replace("'", "").replace("\n", "")
