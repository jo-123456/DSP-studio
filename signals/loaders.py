"""Import real-world sensor data from CSV (Week 4)."""
import pandas as pd


def numeric_columns(file_or_path):
    """Read a CSV and return only its numeric columns (rows with gaps dropped)."""
    try:
        table = pd.read_csv(file_or_path)
    except Exception as e:  # empty file, bad delimiter, ...
        raise ValueError(f"Could not read the CSV: {e}") from e
    nums = table.select_dtypes(include="number").dropna().reset_index(drop=True)
    if nums.shape[1] == 0 or len(nums) < 16:
        raise ValueError("The CSV needs a numeric column with at least 16 rows")
    return nums
