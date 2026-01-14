import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

from src.logger import get_logger

logger = get_logger("src/data/preprocessing.py")


def drop_constant_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Drop features that have constant values across all samples.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        feature_columns: List of feature column names to check

    Returns:
        tuple: (df_train, df_test, features_dropped)
    """
    features_constant_values = df_train[feature_columns].columns[
        df_train[feature_columns].nunique() == 1
    ].tolist()

    logger.info("Following columns have constant values:")
    for feature in features_constant_values:
        logger.info(f"  - {feature}")

    df_train_clean = df_train.drop(features_constant_values, axis=1)
    df_test_clean = df_test.drop(features_constant_values, axis=1)

    return df_train_clean, df_test_clean, features_constant_values


def drop_highly_correlated_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Drop features that are highly correlated with other features.

    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        threshold: Correlation threshold (default: 0.95)

    Returns:
        tuple: (df_train, df_test, features_dropped)
    """
    correlation_matrix = df_train.corr().abs()
    upper_tri = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    high_correlation_features = [
        col for col in upper_tri.columns if any(upper_tri[col] > threshold)
    ]

    logger.info("Highly correlated features are:")
    for feature in high_correlation_features:
        logger.info(f"  - {feature}")

    df_train_clean = df_train.drop(high_correlation_features, axis=1)
    df_test_clean = df_test.drop(high_correlation_features, axis=1)

    return df_train_clean, df_test_clean, high_correlation_features


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_test_capped: pd.Series,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64], MinMaxScaler, MinMaxScaler]:
    """
    Scale features and targets using MinMaxScaler.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        y_test_capped: Capped test targets
        output_path: Optional path to save scalers

    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
                y_test_capped_scaled, scaler_x, scaler_y)
    """
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
    y_test_capped_scaled = scaler_y.transform(y_test_capped.values.reshape(-1, 1))

    # Save scalers if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_x, output_path / 'scaler_x.pkl')
        joblib.dump(scaler_y, output_path / 'scaler_y.pkl')
        logger.info(f"Scalers saved to {output_path}")

    return (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
            y_test_capped_scaled, scaler_x, scaler_y)


def prepare_lstm_data(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_test_RUL: pd.DataFrame,
    final_features: List[str],
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    RUL_MAX: int = 130
) -> Tuple[pd.DataFrame, pd.DataFrame, npt.NDArray[np.float64],
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Prepare scaled data for LSTM models (includes engine column for sequence generation).

    Args:
        df_train: Training dataframe with RUL column
        df_test: Test dataframe
        df_test_RUL: Test RUL values
        final_features: List of selected feature names
        scaler_x: Fitted feature scaler
        scaler_y: Fitted target scaler
        RUL_MAX: Maximum RUL value for capping (default: 130)

    Returns:
        tuple: (X_train_scaled_lstm, X_test_scaled_lstm, y_train_scaled_lstm,
                y_test_scaled_lstm, y_test_capped_scaled_lstm)
    """
    # Prepare train data
    X_train_lstm = df_train[['engine'] + final_features]
    y_train_lstm = df_train['RUL']

    # Prepare test data
    X_test_lstm = df_test[['engine'] + final_features]
    y_test_lstm = df_test_RUL.iloc[:, -1]
    y_test_capped_lstm = np.minimum(y_test_lstm, RUL_MAX)

    # Scale features (but keep engine column unscaled)
    X_train_scaled_lstm = X_train_lstm.copy()
    X_train_scaled_lstm[final_features] = scaler_x.transform(X_train_lstm[final_features])

    X_test_scaled_lstm = X_test_lstm.copy()
    X_test_scaled_lstm[final_features] = scaler_x.transform(X_test_lstm[final_features])

    # Scale targets
    y_train_scaled_lstm = scaler_y.transform(y_train_lstm.values.reshape(-1, 1))
    y_test_scaled_lstm = scaler_y.transform(y_test_lstm.values.reshape(-1, 1))
    y_test_capped_scaled_lstm = scaler_y.transform(y_test_capped_lstm.values.reshape(-1, 1))

    return (X_train_scaled_lstm, X_test_scaled_lstm, y_train_scaled_lstm,
            y_test_scaled_lstm, y_test_capped_scaled_lstm)


def save_preprocessed_data(
    output_path: Union[str, Path],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_test_capped: pd.Series,
    X_train_scaled: npt.NDArray[np.float64],
    X_test_scaled: npt.NDArray[np.float64],
    y_train_scaled: npt.NDArray[np.float64],
    y_test_scaled: npt.NDArray[np.float64],
    y_test_capped_scaled: npt.NDArray[np.float64],
    X_train_scaled_lstm: Optional[pd.DataFrame] = None,
    X_test_scaled_lstm: Optional[pd.DataFrame] = None,
    y_train_scaled_lstm: Optional[npt.NDArray[np.float64]] = None,
    y_test_scaled_lstm: Optional[npt.NDArray[np.float64]] = None,
    y_test_capped_scaled_lstm: Optional[npt.NDArray[np.float64]] = None,
    df_train_scaled_lstm_with_error_vector: Optional[pd.DataFrame] = None
) -> None:
    """
    Save all preprocessed data to disk.

    Args:
        output_path: Directory to save processed data
        X_train, X_test, y_train, y_test, y_test_capped: Unscaled data
        X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
        y_test_capped_scaled: Scaled data
        X_train_scaled_lstm, X_test_scaled_lstm, y_train_scaled_lstm,
        y_test_scaled_lstm, y_test_capped_scaled_lstm: LSTM-specific data (optional)
        df_train_scaled_lstm_with_error_vector: LSTM data with error vectors (optional)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save unscaled data
    np.save(output_path / 'X_train.npy', X_train.values)
    np.save(output_path / 'X_test.npy', X_test.values)
    np.save(output_path / 'y_train.npy', y_train.values)
    np.save(output_path / 'y_test.npy', y_test.values)
    np.save(output_path / 'y_test_capped.npy', y_test_capped.values)

    # Save scaled data
    np.save(output_path / 'X_train_scaled.npy', X_train_scaled)
    np.save(output_path / 'X_test_scaled.npy', X_test_scaled)
    np.save(output_path / 'y_train_scaled.npy', y_train_scaled)
    np.save(output_path / 'y_test_scaled.npy', y_test_scaled)
    np.save(output_path / 'y_test_capped_scaled.npy', y_test_capped_scaled)

    # Save LSTM-specific data if provided
    if X_train_scaled_lstm is not None:
        X_train_scaled_lstm.to_parquet(
            output_path / 'X_train_scaled_lstm.parquet',
            engine='pyarrow',
            compression='snappy'
        )

    if X_test_scaled_lstm is not None:
        X_test_scaled_lstm.to_parquet(
            output_path / 'X_test_scaled_lstm.parquet',
            engine='pyarrow',
            compression='snappy'
        )

    if y_train_scaled_lstm is not None:
        np.save(output_path / 'y_train_scaled_lstm.npy', y_train_scaled_lstm)

    if y_test_scaled_lstm is not None:
        np.save(output_path / 'y_test_scaled_lstm.npy', y_test_scaled_lstm)

    if y_test_capped_scaled_lstm is not None:
        np.save(output_path / 'y_test_capped_scaled_lstm.npy', y_test_capped_scaled_lstm)

    if df_train_scaled_lstm_with_error_vector is not None:
        df_train_scaled_lstm_with_error_vector.to_parquet(
            output_path / 'df_train_scaled_lstm_with_error_vector.parquet',
            engine='pyarrow',
            compression='snappy'
        )

    logger.info(f"Preprocessed data saved to {output_path}")
