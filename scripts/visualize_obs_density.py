"""
Visualize observation density maps for different sampling scenarios.
Shows how frequently each location is included in the training set.
Delegates to da_stdk.viz.obs_density.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from da_stdk.viz import plot_observation_density_maps


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2x2 observation density maps (Fixed/Random × Uniform/Clustered)."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to CSV (e.g. data/2b/2b_8_train.csv). Default: data/2b/2b_8_train.csv",
    )
    parser.add_argument("--obs_ratio", type=float, default=0.1, help="Observation ratio")
    parser.add_argument("--intensity", type=float, default=10.0, help="Corner pattern intensity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figure. Default: results/",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "2b"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path or (data_dir / "2b_8_train.csv")
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    save_path = output_dir / f"obs_density_2b8_ratio{args.obs_ratio}.png"

    print("=" * 60)
    print("Observation Density Visualization")
    print("=" * 60)

    fig = plot_observation_density_maps(
        data_path=data_path,
        obs_ratio=args.obs_ratio,
        intensity=args.intensity,
        seed=args.seed,
        save_path=save_path,
    )

    print("\nVisualization complete!")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
