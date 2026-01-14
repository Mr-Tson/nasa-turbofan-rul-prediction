"""
PyTorch Dataset classes for NASA CMAPSS turbofan engine data.
"""
from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Dataset for vanilla LSTM model.
    Creates sequences of fixed window size from time series data.
    """

    def __init__(self, df: pd.DataFrame, feature_names: List[str], window_size: int = 30) -> None:
        """
        Args:
            df: DataFrame with 'engine' column and feature columns
            feature_names: List of feature column names to use
            window_size: Length of sequence window (default: 30)
        """
        self.df = df
        self.feature_names = feature_names
        self.window_size = window_size
        self.sequences, self.targets = self._create_sequences()

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences and targets from the dataframe."""
        sequences = []
        targets = []

        for engine_id in self.df['engine'].unique():
            engine_data = self.df[self.df['engine'] == engine_id]
            features = engine_data[self.feature_names].values
            rul_values = engine_data['RUL'].values

            if len(features) >= self.window_size:
                for idx in range(len(features) - self.window_size + 1):
                    sequences.append(features[idx:idx + self.window_size])
                    targets.append(rul_values[idx + self.window_size - 1])

        return (
            torch.tensor(np.array(sequences), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        item["seq"] = self.sequences[idx]
        item["target"] = self.targets[idx]
        return item


class TestSequenceDataset(Dataset):
    """
    Dataset for test data with vanilla LSTM model.
    Only predicts the last timestep for each engine.
    """

    def __init__(
        self,
        X_test: pd.DataFrame,
        feature_names: List[str],
        y_test: npt.NDArray[np.float64],
        window_size: int = 30
    ) -> None:
        """
        Args:
            X_test: Test DataFrame with 'engine' column and feature columns
            feature_names: List of feature column names to use
            y_test: Test targets (RUL values)
            window_size: Length of sequence window (default: 30)
        """
        self.X_test = X_test
        self.feature_names = feature_names
        self.y_test = y_test
        self.window_size = window_size
        self.sequences, self.targets = self._create_sequences()

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences and targets from the test dataframe."""
        sequences = []
        targets = []

        for i, engine_id in enumerate(self.X_test['engine'].unique()):
            engine_data = self.X_test[self.X_test['engine'] == engine_id]
            features = engine_data[self.feature_names].values

            true_rul = self.y_test[i, 0]

            if len(features) >= self.window_size:
                sequence = features[-self.window_size:]
            else:
                # Left pad with zeros if not enough data
                padding_needed = self.window_size - len(features)
                padding = np.zeros((padding_needed, features.shape[1]), dtype=features.dtype)
                sequence = np.concatenate((padding, features), axis=0)

            sequences.append(sequence)
            targets.append(true_rul)

        return (
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        item["seq"] = self.sequences[idx]
        item["target"] = self.targets[idx]
        return item


class SequenceShiftedDataset(Dataset):
    """
    Dataset for LSTM with auxiliary sensor feature prediction.
    Creates sequences and shifted sequences for predicting future sensor states.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        window_size: int = 30,
        lookahead: int = 10,
        noise_std: float = 0.01
    ) -> None:
        """
        Args:
            df: DataFrame with 'engine', 'cycle', feature columns, and 'RUL' column
            feature_names: List of feature column names to use
            window_size: Length of sequence window (default: 30)
            lookahead: Number of timesteps to look ahead (default: 10)
            noise_std: Standard deviation of noise to add to failure vector (default: 0.01)
        """
        self.df = df
        self.feature_names = feature_names
        self.window_size = window_size
        self.lookahead = lookahead
        self.noise_std = noise_std
        self.sequences, self.sequences_shifted, self.targets = self._create_sequences()

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create sequences, shifted sequences, and targets."""
        sequences = []
        sequences_shifted = []
        targets = []

        for engine_id in self.df['engine'].unique():
            # Extract components
            engine_data = self.df[self.df['engine'] == engine_id]
            features = engine_data[self.feature_names].values
            cycle_values = engine_data['cycle'].values
            rul_values = engine_data['RUL'].values

            # Get failure vector (last row represents failure state)
            failure_vector = features[-1:]

            # Temporal extension for cycles
            cycle_diff = cycle_values[-1] - cycle_values[-2] if len(cycle_values) > 1 else 0.01
            extension_cycles = np.array([
                cycle_values[-1] + (i + 1) * cycle_diff for i in range(self.lookahead)
            ])

            # Extension for features with noise
            extension_noise = np.random.normal(0, self.noise_std, (self.lookahead, features.shape[1]))
            extension_features = np.tile(failure_vector, (self.lookahead, 1)) + extension_noise
            extension_features = np.clip(extension_features, 0.0, 1.0)

            # Full extended features
            full_features_ext = np.vstack([features, extension_features])

            # Replace cycles in extended features
            if 'cycle' in self.feature_names:
                cycle_idx = self.feature_names.index('cycle')
                full_features_ext[len(features):, cycle_idx] = extension_cycles

            # Extend RUL
            extension_rul = np.zeros(self.lookahead)
            rul_ext = np.concatenate([rul_values, extension_rul])

            # Create sequences
            if len(features) >= self.window_size:
                for idx in range(len(features) - self.window_size + 1):
                    sequences.append(full_features_ext[idx:idx + self.window_size])
                    sequences_shifted.append(
                        full_features_ext[idx + self.lookahead:idx + self.lookahead + self.window_size]
                    )
                    targets.append(rul_ext[idx + self.window_size - 1])

        return (
            torch.tensor(np.array(sequences), dtype=torch.float32),
            torch.tensor(np.array(sequences_shifted), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.float32)
        )

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {}
        item["seq"] = self.sequences[idx]
        item["seq_shifted"] = self.sequences_shifted[idx]
        item["target"] = self.targets[idx]
        return item


def collate_fn_vanilla(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for vanilla LSTM datasets.

    Args:
        batch: List of items from SequenceDataset or TestSequenceDataset

    Returns:
        tuple: (sequences, targets)
    """
    sequences = torch.stack([item['seq'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return sequences, targets


def collate_fn_shifted(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for shifted sequence datasets.

    Args:
        batch: List of items from SequenceShiftedDataset

    Returns:
        tuple: (sequences, sequences_shifted, targets)
    """
    sequences = torch.stack([item['seq'] for item in batch])
    sequences_shifted = torch.stack([item['seq_shifted'] for item in batch])
    targets = torch.stack([item['target'] for item in batch])
    return sequences, sequences_shifted, targets
