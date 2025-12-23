from src.data.load_data import load_cmapss_data
from src.visualization.plots import (
    plot_histogram_features_distribution,
    plot_boxplot_features_distribution,
    plot_correlation_features,
    plot_features_subset_engines,
    plot_features_all_engines
)
from src.config import OUTPUT_PATH

import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load data
    df_train = load_cmapss_data("FD001", "train")

    # Generate plots
    fig1 = plot_histogram_features_distribution(df_train)
    fig1.savefig(
        OUTPUT_PATH / 'figures' / 'features_distribution_histogram.png',
        bbox_inches='tight',
        dpi=300
        )

    fig2 = plot_boxplot_features_distribution(df_train)
    fig2.savefig(
        OUTPUT_PATH / 'figures' / 'features_distribution_boxplot.png',
        bbox_inches='tight',
        dpi=300        
        )

    fig3 = plot_correlation_features(df_train)
    fig3.savefig(
        OUTPUT_PATH / 'figures' / 'features_correlation.png',
        bbox_inches='tight',
        dpi=300        
        )

    fig4 = plot_features_subset_engines(df_train, 10)
    fig4.savefig(
        OUTPUT_PATH / 'figures' / 'features_over_time_10_engines.png',
        bbox_inches='tight',
        dpi=300        
        )

    fig5 = plot_features_all_engines(df_train)
    fig5.savefig(
        OUTPUT_PATH / 'figures' / 'features_over_time_all_engines.png',
        bbox_inches='tight',
        dpi=300        
        )

if __name__ == "__main__":
    main()
