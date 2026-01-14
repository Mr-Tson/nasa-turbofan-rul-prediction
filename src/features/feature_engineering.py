from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.logger import get_logger

logger = get_logger(__name__)


def calculate_rul(df_train: pd.DataFrame, max_rul: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine in the training data.

    Args:
        df_train: Training dataframe with 'engine' and 'cycle' columns
        max_rul: Optional maximum RUL value for capping (e.g., 130)

    Returns:
        pd.DataFrame: Training dataframe with 'RUL' column added
    """
    # Calculate end of life for each engine
    df_train_RUL = df_train.groupby('engine').agg({'cycle': 'max'})
    df_train_RUL.rename(columns={'cycle': 'end_of_life'}, inplace=True)

    # Merge and calculate RUL
    df_train_with_rul = pd.merge(df_train, df_train_RUL, how='left', on=['engine'])
    df_train_with_rul['RUL'] = df_train_with_rul['end_of_life'] - df_train_with_rul['cycle']
    df_train_with_rul.drop(['end_of_life'], axis=1, inplace=True)

    # Apply RUL capping if specified
    if max_rul is not None:
        df_train_with_rul.loc[df_train_with_rul['RUL'] > max_rul, 'RUL'] = max_rul

    return df_train_with_rul


def backward_regression(
    X: pd.DataFrame,
    y: pd.Series,
    initial_list: Optional[List[str]] = None,
    threshold_out: float = 0.05,
    verbose: bool = True
) -> List[str]:
    """
    Perform backward elimination regression to select features.

    Args:
        X: Feature values (pd.DataFrame)
        y: Target variable (pd.Series)
        initial_list: Features header (not used, kept for compatibility)
        threshold_out: p-value threshold of features to drop
        verbose: True to produce lots of logging output

    Returns:
        list: Selected features for modeling
    """
    included = list(X.columns)
    while True:
        changed = False
        model = sm.OLS(y, sm.add_constant(X[included])).fit()
        pvalues = model.pvalues.iloc[1:]  # do not include intercept beta_0
        worst_pval = pvalues.max()  # null if pvalues is empty
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                logger.info(f"Removed feature: {worst_feature} (p-value: {worst_pval:.4f})")
        if not changed:
            break
    return included


def extract_last_cycle_from_test(df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last cycle data for each engine from the test dataset.

    Args:
        df_test: Test dataframe with 'engine' and 'cycle' columns

    Returns:
        pd.DataFrame: Test dataframe containing only the last cycle for each engine
    """
    df_test_max = df_test[
        df_test['cycle'] == df_test.groupby(['engine'])['cycle'].transform('max')
    ]
    return df_test_max


def analyze_feature_trends(df_train: pd.DataFrame, final_features: List[str]) -> Dict[str, str]:
    """
    Analyze feature trends with respect to cycle for each engine.
    Determines if features have upward, downward, or no clear trend.

    Args:
        df_train: Training dataframe with 'engine', 'cycle' columns and RUL
        final_features: List of feature names to analyze

    Returns:
        dict: Dictionary mapping feature names to trend labels
              ('upward_trend', 'downward_trend', 'no_clear_trend')
    """
    # Calculate correlation between cycle and each feature for each engine
    grouped_corr = df_train.groupby('engine').apply(
        lambda x: x[final_features].corr()['cycle'],
        include_groups=False
    )

    # Determine trend based on correlation sign consistency across all engines
    is_upward = (grouped_corr > 0).all()
    is_downward = (grouped_corr < 0).all()

    labels = np.select(
        [is_upward, is_downward],
        ['upward_trend', 'downward_trend'],
        default='no_clear_trend'
    )

    # Create a summary DataFrame
    summary_trend = pd.DataFrame([labels], columns=grouped_corr.columns, index=['Trend']).T

    # Sort by trend
    trend_order = ['upward_trend', 'downward_trend', 'no_clear_trend']
    summary_trend['Trend'] = pd.Categorical(
        summary_trend['Trend'], categories=trend_order, ordered=True
    )
    summary_trend_sorted = summary_trend.sort_values(by='Trend')

    logger.info("Feature trend analysis summary:\n%s", summary_trend_sorted)

    return summary_trend_sorted.to_dict()['Trend']


def add_feature_error_vector(
    engine_id: Union[int, float],
    df_train_scaled: pd.DataFrame,
    summary_trend_dict: Dict[str, str]
) -> pd.DataFrame:
    """
    Add a feature error vector (failure state) for a single engine.

    This creates a synthetic "failure" state by extrapolating one cycle beyond
    the last observed cycle and setting feature values based on their trends:
    - upward_trend features → 1.0 (maximum failure)
    - downward_trend features → 0.0 (minimum failure)
    - no_clear_trend features → mean value

    Args:
        engine_id: Engine ID to process
        df_train_scaled: Scaled training dataframe with 'engine' and 'cycle' columns
        summary_trend_dict: Dictionary mapping feature names to trends

    Returns:
        pd.DataFrame: Engine data with error vector appended as the last row
    """
    df_train_scaled_engine = df_train_scaled[df_train_scaled['engine'] == engine_id]

    feature_error_data = pd.Series(index=df_train_scaled_engine.columns, dtype=float)
    feature_error_data['engine'] = float(engine_id)

    # Extrapolate cycle value
    if len(df_train_scaled_engine) >= 2:
        scaled_cycle_val_last = df_train_scaled_engine['cycle'].iloc[-1]
        scaled_cycle_val_second_last = df_train_scaled_engine['cycle'].iloc[-2]
        scaled_cycle_diff = scaled_cycle_val_last - scaled_cycle_val_second_last
        feature_error_data['cycle'] = scaled_cycle_val_last + scaled_cycle_diff
    else:
        feature_error_data['cycle'] = df_train_scaled_engine['cycle'].iloc[-1]

    feature_error_data['RUL'] = 0.0  # RUL for failure state is 0

    # Handle other feature columns based on trend
    for col in df_train_scaled_engine.columns:
        if col not in ['engine', 'cycle', 'RUL']:
            trend = summary_trend_dict.get(col)
            if trend == 'upward_trend':
                feature_error_data[col] = 1.0
            elif trend == 'downward_trend':
                feature_error_data[col] = 0.0
            else:  # 'no_clear_trend' or not found in dict
                feature_error_data[col] = df_train_scaled_engine[col].mean()

    # Append error vector to engine data
    feature_error_df = pd.DataFrame([feature_error_data])
    df_train_scaled_engine_with_error_vector = pd.concat(
        [df_train_scaled_engine, feature_error_df], ignore_index=True
    )

    return df_train_scaled_engine_with_error_vector


def create_df_with_feature_errors(
    df_train_scaled: pd.DataFrame,
    summary_trend_dict: Dict[str, str]
) -> pd.DataFrame:
    """
    Create dataframe with feature error vectors for all engines.

    Args:
        df_train_scaled: Scaled training dataframe
        summary_trend_dict: Dictionary mapping feature names to trends

    Returns:
        pd.DataFrame: Complete dataframe with error vectors appended for each engine
    """
    df_engines_with_error_vectors = []
    for engine_id in df_train_scaled['engine'].unique():
        df_train_scaled_engine_with_error_vector = add_feature_error_vector(
            engine_id, df_train_scaled, summary_trend_dict
        )
        df_engines_with_error_vectors.append(df_train_scaled_engine_with_error_vector)

    return pd.concat(df_engines_with_error_vectors, ignore_index=True)
