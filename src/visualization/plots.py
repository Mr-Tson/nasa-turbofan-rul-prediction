import math
import random

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from typing import List, Tuple, Any


def _create_subplot_grid(num_items: int, cols_per_row: int, figsize_per_col: int = 5,
                         figsize_per_row: int = 4, sharex: bool = False) -> Tuple[Figure, np.ndarray]:
    """
    Helper function to create a grid of subplots.

    Args:
        num_items: Number of items to plot
        cols_per_row: Number of columns per row in the grid
        figsize_per_col: Width per column in inches
        figsize_per_row: Height per row in inches
        sharex: Whether subplots should share x-axis

    Returns:
        Tuple of (figure, flattened_axes_array)
    """
    num_rows = math.ceil(num_items / cols_per_row)
    fig, axes = plt.subplots(
        num_rows, cols_per_row,
        figsize=(cols_per_row * figsize_per_col, num_rows * figsize_per_row),
        sharex=sharex
    )
    axes = axes.flatten()
    return fig, axes


def plot_histogram_features_distribution(df: pd.DataFrame) -> Figure:
    """
    Plot histograms with KDE for all features in the dataframe.

    Args:
        df: DataFrame containing features to plot
    
    Returns:
        the plot figure
    """
    num_cols = len(df.columns)
    num_cols_per_row = 5

    fig, axes = _create_subplot_grid(num_cols, num_cols_per_row)

    for i, col in enumerate(df.columns):
        # get histogram data to scale y-axis correctly
        counts, bins = np.histogram(df[col], bins=30)
        max_count = counts.max()

        sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Histogram of {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].set_ylim(0, max_count * 1.15)

    # delete remaining axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribution of all features", fontsize=16, y=1.0)
    plt.tight_layout()
    return fig

def plot_boxplot_features_distribution(df: pd.DataFrame) -> Figure:
    """
    Plot boxplots for all features in the dataframe to visualize outliers and distribution.

    Args:
        df: DataFrame containing features to plot
    
    Returns:
        the plot figure        
    """
    num_cols = len(df.columns)
    num_cols_per_row = 5

    fig, axes = _create_subplot_grid(num_cols, num_cols_per_row)

    for i, col in enumerate(df.columns):
        axes[i].boxplot(df[col])
        axes[i].set_title(f"{col}")

    # delete remaining axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribution of all features", fontsize=16, y=1.0)
    plt.tight_layout()
    return fig

def plot_correlation_features(df: pd.DataFrame) -> Figure:
    """
    Plot a correlation heatmap for all features in the dataframe.

    Args:
        df: DataFrame containing features to analyze
    
    Returns:
        the plot figure        
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(df.corr().fillna(0), annot=True, fmt='.2f', cmap='RdYlGn', linewidths=0.2, yticklabels=True)
    ax.set_title("Correlation heatmap for all features", fontsize=16, pad=20)
    plt.tight_layout()
    return fig

def generate_random_hex_colors(num_colors: int) -> List[str]:
    """
    Generate a list of random hex color codes.

    Args:
        num_colors: Number of random colors to generate

    Returns:
        List of hex color codes as strings
    """
    colors = []
    for _ in range(num_colors):
        # Generate a random hex color code
        hex_color_code = '#%06x' % random.randint(0, 0xFFFFFF)
        colors.append(hex_color_code)
    return colors

def _plot_features_engines_internal(
    df: pd.DataFrame, engines_to_plot: List[int], title: str, y_box_anchor: float, n_cols_legend: int) -> Figure:
    """
    Internal helper function to plot features for engines

    Args:
        df: DataFrame containing features to plot
        engines_to_plot: A list containing the engine ids to plot
        title: Title for plot
        y_box_anchor: y-box-anchor value for legend in plot
        n_cols_legend: number of columns for legend
    Returns:
        the plot figure
    """
    num_cols = len(df.columns[1:])
    num_cols_per_row = 2
    plot_items = list(df.columns)[1:]
    fig, ax = _create_subplot_grid(num_cols, num_cols_per_row, figsize_per_col=10,
                                     figsize_per_row=5, sharex=True)

    random_colors = generate_random_hex_colors(len(engines_to_plot))

    # Create custom legend handles
    legend_elements = [Patch(facecolor=random_colors[i], label=f'Engine {engines_to_plot[i]}')
                    for i in range(len(engines_to_plot))]

    for engine_idx, engine in enumerate(engines_to_plot):
        for idx, item in enumerate(plot_items):
            f = sns.lineplot(data=df[df['engine'] == engine],
                            x='cycle', y=item, color=random_colors[engine_idx],
                            linewidth=1.0, ax=ax[idx])

    for ax_idx in range(idx+1, len(ax)):
        fig.delaxes(ax[ax_idx])

    fig.suptitle(title, fontsize=20, fontweight='bold', y=1.0)
    fig.legend(handles=legend_elements, loc='lower center', frameon=False, bbox_to_anchor=(0.5, y_box_anchor),
            ncol=n_cols_legend, fontsize=10)
    plt.tight_layout()
    return fig

def plot_features_subset_engines(df: pd.DataFrame, number_of_engines_to_plot: int) -> Figure:
    """
    Plot sensor features over time for a random subset of engines.

    Args:
        df: DataFrame containing features to plot
        number_of_engines_to_plot: Number of random engines to include in the plot
    
    Returns:
        the plot figure        
    """
    engines = sorted(list(df['engine'].unique()))
    number_of_engines_to_plot = min(number_of_engines_to_plot, len(engines))
    random_engines = random.sample(engines, k=number_of_engines_to_plot)
    fig = _plot_features_engines_internal(df, random_engines, f"Sensor plot for {len(random_engines)} random engines", -0.02, 5)
    return fig

def plot_features_all_engines(df: pd.DataFrame) -> Figure:
    """
    Plot sensor features over time for all engines in the dataset.

    Args:
        df: DataFrame containing features to plot
    
    Returns:
        the plot figure        
    """    
    engines = sorted(list(df['engine'].unique()))
    fig = _plot_features_engines_internal(df, engines, f"Sensor plot for all {len(engines)} engines", -0.055, 10)
    return fig

    
