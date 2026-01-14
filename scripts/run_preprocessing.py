import numpy as np
import pandas as pd
from pathlib import Path

from src.data.load_data import load_cmapss_data, load_test_rul_cmapss_data
from src.data.preprocessing import (
    drop_constant_features,
    drop_highly_correlated_features,
    scale_features,
    prepare_lstm_data,
    save_preprocessed_data
)
from src.features.feature_engineering import (
    calculate_rul,
    backward_regression,
    extract_last_cycle_from_test,
    analyze_feature_trends,
    create_df_with_feature_errors
)
from src.config import setting_names, sensor_names, RUL_MAX, PROCESSED_DATA_PATH, DATASET
from src.logger import get_logger

# Get script path relative to project root
project_root = Path(__file__).resolve().parent.parent
script_path = Path(__file__).resolve().relative_to(project_root)
logger = get_logger(str(script_path))


def perform_preprocessing():
    """Run the complete preprocessing pipeline."""
    logger.info("=" * 80)
    logger.info("NASA CMAPSS Data Preprocessing Pipeline")
    logger.info("=" * 80)

    # Step 1: Load data
    logger.info("Step 1 - Loading data...")
    df_train = load_cmapss_data(DATASET, 'train')
    df_test = load_cmapss_data(DATASET, 'test')
    df_test_RUL = load_test_rul_cmapss_data(DATASET)
    logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}")

    # Step 2: Check for NaN values
    logger.info("Step 2 - Checking for NaN values...")
    nan_count = df_train.isna().sum().sum()
    logger.info(f"Total NaN values in training data: {nan_count}")

    # Step 3: Drop constant features
    logger.info("Step 3 - Dropping constant features...")
    all_features = list(setting_names + sensor_names)
    df_train, df_test, constant_features = drop_constant_features(
        df_train, df_test, all_features
    )
    logger.info(f"Dropped {len(constant_features)} constant features")

    # Step 4: Drop highly correlated features
    logger.info("Step 4 -  Dropping highly correlated features...")
    df_train, df_test, correlated_features = drop_highly_correlated_features(
        df_train, df_test, threshold=0.95
    )
    logger.info(f"Dropped {len(correlated_features)} highly correlated features")

    # Step 5: Calculate RUL
    logger.info("Step 5 - Calculating RUL...")
    df_train = calculate_rul(df_train, max_rul=RUL_MAX)
    logger.info(f"RUL calculated with max cap at {RUL_MAX}")

    # Step 6: Feature selection using backward regression
    logger.info("Step 6 - Performing backward regression for feature selection...")
    X = df_train.iloc[:, 1:-1]  # Exclude 'engine' and 'RUL'
    y = df_train.iloc[:, -1]    # RUL column
    final_features = backward_regression(X, y, threshold_out=0.05, verbose=True)
    logger.info(f"Selected {len(final_features)} features: {final_features}")

    # Step 7: Extract last cycle from test data
    logger.info("Step 7 - Extracting last cycle from test data...")
    df_test_max = extract_last_cycle_from_test(df_test)
    logger.info(f"Extracted {len(df_test_max)} test samples (one per engine)")

    # Step 8: Prepare train and test data
    logger.info("Step 8 - Preparing and scaling data...")
    X_train = df_train[final_features]
    y_train = df_train['RUL']
    X_test = df_test_max[final_features]
    y_test = df_test_RUL.iloc[:, -1]
    y_test_capped = np.minimum(y_test, RUL_MAX)

    # Scale features and targets
    (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled,
     y_test_capped_scaled, scaler_x, scaler_y) = scale_features(
        X_train, X_test, y_train, y_test, y_test_capped,
        output_path=PROCESSED_DATA_PATH
    )
    logger.info("Features and targets scaled using MinMaxScaler")

    # Prepare LSTM-specific data
    logger.info("Preparing LSTM-specific data...")
    (X_train_scaled_lstm, X_test_scaled_lstm, y_train_scaled_lstm,
     y_test_scaled_lstm, y_test_capped_scaled_lstm) = prepare_lstm_data(
        df_train, df_test, df_test_RUL, final_features, scaler_x, scaler_y, RUL_MAX
    )
    logger.info("LSTM data prepared (includes engine column for sequence generation)")

    # Step 9: Analyze feature trends and create error vectors
    logger.info("Step 9 - Analyzing feature trends and creating error vectors...")
    summary_trend_dict = analyze_feature_trends(df_train, final_features)

    # Create LSTM dataframe with error vectors
    y_train_scaled_lstm_df = pd.DataFrame(y_train_scaled_lstm, columns=['RUL'])
    df_train_scaled_lstm = pd.concat([X_train_scaled_lstm, y_train_scaled_lstm_df], axis=1)

    df_train_scaled_lstm_with_error_vector = create_df_with_feature_errors(
        df_train_scaled_lstm, summary_trend_dict
    )
    logger.info("Error vectors created for LSTM auxiliary sensor prediction")

    # Step 10: Save all preprocessed data
    logger.info("Step 10 - Saving preprocessed data...")
    save_preprocessed_data(
        output_path=PROCESSED_DATA_PATH,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_test_capped=y_test_capped,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train_scaled=y_train_scaled,
        y_test_scaled=y_test_scaled,
        y_test_capped_scaled=y_test_capped_scaled,
        X_train_scaled_lstm=X_train_scaled_lstm,
        X_test_scaled_lstm=X_test_scaled_lstm,
        y_train_scaled_lstm=y_train_scaled_lstm,
        y_test_scaled_lstm=y_test_scaled_lstm,
        y_test_capped_scaled_lstm=y_test_capped_scaled_lstm,
        df_train_scaled_lstm_with_error_vector=df_train_scaled_lstm_with_error_vector
    )

    logger.info("Preprocessing complete!")
    # TODO: Make a small summary what was done in this script.

if __name__ == "__main__":
    perform_preprocessing()
