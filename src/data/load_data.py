import pandas as pd
from pathlib import Path
from src.config import DATA_PATH, col_names

def load_cmapss_data(dataset_name: str, dataset_type: str) -> pd.DataFrame:
    """
    Load NASA C-MAPSS training or test data into a DataFrame.

    Args:
        dataset_name: Dataset identifier (e.g., 'FD001', 'FD002', 'FD003', 'FD004')
        dataset_type: Dataset type ('train' or 'test')

    Returns:
        DataFrame containing engine operational settings and feature sensors

    Raises:
        FileNotFoundError: If the dataset file doesn't exist at the expected path
    """
    filepath = DATA_PATH / f'{dataset_type}_{dataset_name}.txt'
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath, sep='\s+', header=None, names=col_names)
    return df

def load_test_rul_cmapss_data(dataset_name: str) -> pd.DataFrame:
    """
    Load RUL for test data into a dataframe

    Args:
        dataset_name: Dataset identifier (e.g., 'FD001', 'FD002', 'FD003', 'FD004')

    Returns:
        DataFrame containing test engines RUL

    Raises:
        FileNotFoundError: If the dataset file doesn't exist at the expected path
    """
    filepath = DATA_PATH / f'RUL_{dataset_name}.txt'
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
    df_test_RUL = pd.read_csv(filepath, sep='\s+', header=None, names=['RUL'])
    return df_test_RUL


