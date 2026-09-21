"""
data_loader.py
Handles loading and basic validation of the product catalog.
"""

import os
import pandas as pd

REQUIRED_COLUMNS = {"id", "name", "category", "description"}


def load_products(csv_path: str) -> pd.DataFrame:
    """
    Load the product catalog from a CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the products CSV file.

    Returns:
        A pandas DataFrame containing the product catalog.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Product catalog not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Basic cleanup
    df["description"] = df["description"].fillna("")
    df["category"] = df["category"].fillna("Unknown")
    df = df.drop_duplicates(subset="id").reset_index(drop=True)

    return df


def get_default_path() -> str:
    """Return the default path to the bundled sample dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "data", "products.csv")
