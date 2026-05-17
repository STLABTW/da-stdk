"""
Visualization helpers for training curves, observation patterns, spatial and temporal plots.
"""

from .basis import plot_basis_evolution
from .kaust_analysis import (
    build_summary_by_method,
    create_cluster_gain_plot,
    create_coverage_4methods_plot,
    create_coverage_comparison_plot,
    create_paper_main_figures,
    create_spatial_coverage_comparison_plots,
    create_table_4_4,
    load_table_4_4_results,
    print_table_4_4,
)
from .obs import create_observation_density_map, plot_observation_pattern
from .obs_density import compute_observation_density, plot_observation_density_maps
from .predictions import plot_predictions
from .spatial import (
    create_averaged_spatial_coverage,
    plot_spatial_coverage_and_qhat,
    plot_spatial_mse,
)
from .temporal import plot_temporal_series, plot_temporal_series_conformal_highlight
from .training import plot_training_curves

__all__ = [
    "plot_training_curves",
    "plot_observation_pattern",
    "create_observation_density_map",
    "compute_observation_density",
    "plot_observation_density_maps",
    "plot_predictions",
    "plot_basis_evolution",
    "plot_spatial_mse",
    "plot_spatial_coverage_and_qhat",
    "create_averaged_spatial_coverage",
    "plot_temporal_series",
    "plot_temporal_series_conformal_highlight",
    "load_table_4_4_results",
    "create_table_4_4",
    "build_summary_by_method",
    "create_cluster_gain_plot",
    "create_coverage_4methods_plot",
    "create_coverage_comparison_plot",
    "create_spatial_coverage_comparison_plots",
    "create_paper_main_figures",
    "print_table_4_4",
]
