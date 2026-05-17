#!/usr/bin/env python3
"""
Run KAUST data experiment: all scenarios × models with N replicates each.

Scenarios: Fixed_Uniform, Fixed_Clustered, Random_Uniform, Random_Clustered
Models: STDK, DA-STDK
Output: results/kaust_data_<timestamp>/<Scenario>_<Model>/ (each with experiments/1..N)

Forces: split_method=random, calibration_ratio_from_train=0.2, calibration_split_method=random.
With train_ratio=0.8, calibration uses validation split.

Non-crossing: λ (non_crossing_lambda) is tuned via grid search (scripts/grid_search_non_crossing_lambda.py,
e.g. 0~0.1 log scale); then pass --non_crossing_lambda to this script for the KAUST experiment.
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Scenario: (obs_method, obs_spatial_pattern)
SCENARIOS = {
    "Fixed_Uniform": ("site-wise", "uniform"),
    "Fixed_Clustered": ("site-wise", "corner"),
    "Random_Uniform": ("random", "uniform"),
    "Random_Clustered": ("random", "corner"),
}
# Model: (spatial_learnable, spatial_init_method)
# Paper: DA-STDK uses balanced k-means for spatial init
MODELS = {
    "STDK": (False, "uniform"),
    "DA-STDK": (True, "kmeans_balanced"),
}


def main():
    ap = argparse.ArgumentParser(
        description="Run KAUST data experiment (all scenarios × models, N replicates)."
    )
    ap.add_argument(
        "--config", type=str, default="configs/config_default.yaml", help="Base config YAML"
    )
    ap.add_argument("--n_experiments", type=int, default=10, help="Replicates per scenario×model")
    ap.add_argument(
        "--data_file",
        type=str,
        default="data/2b/2b_8.csv",
        help="Data file (KAUST experiment uses 2b_8)",
    )
    ap.add_argument("--base_seed", type=int, default=2025)
    ap.add_argument(
        "--parallel", action="store_true", help="Run replicates in parallel per scenario×model"
    )
    ap.add_argument("--dry-run", action="store_true", help="Print commands only")
    ap.add_argument(
        "--analyze", action="store_true", help="Run analyze_kaust_results.py after all runs finish"
    )
    ap.add_argument(
        "--paper_main_only",
        action="store_true",
        help="When used with --analyze, produce only the 3 main paper figures (fig_main_coverage_4methods, fig_main_cluster_gain, fig_main_spatial_delta_clustered). "
        "Without this flag, analysis still produces fig_main_coverage_4methods plus coverage_stdk_nominal_vs_dastdk_cluster and spatial_coverage_compare_*, but not cluster_gain or spatial_delta_clustered.",
    )
    ap.add_argument(
        "--skip_coverage_plot",
        action="store_true",
        help="When used with --analyze, skip STDK-vs-DA coverage comparison plot",
    )
    ap.add_argument(
        "--skip_spatial_compare_plot",
        action="store_true",
        help="When used with --analyze, skip spatial map comparison plots",
    )
    # δ reparameterization + P_nc(δ): default on for DA-STDK only (STDK never uses δ)
    ap.add_argument(
        "--use_delta_reparameterization",
        action="store_true",
        default=True,
        help="Use δ reparameterization + P_nc(δ) for DA-STDK (default: True)",
    )
    ap.add_argument(
        "--no_delta_reparameterization",
        action="store_true",
        help="Disable δ for DA-STDK (use base config)",
    )
    ap.add_argument(
        "--non_crossing_lambda",
        type=float,
        default=0.0,
        help="P_nc(δ) penalty weight λ for DA-STDK only (tuned via grid_search_non_crossing_lambda.py). Default: 0.0",
    )
    args = ap.parse_args()
    if args.no_delta_reparameterization:
        args.use_delta_reparameterization = False

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "train_default.py"
    analyze_script = repo_root / "scripts" / "analyze_kaust_results.py"
    if not train_script.exists():
        train_script = repo_root / "train_default.py"
    if not train_script.exists():
        print(f"Not found: {train_script}", file=sys.stderr)
        sys.exit(1)

    base_config_path = repo_root / args.config
    if not base_config_path.exists():
        print(f"Config not found: {base_config_path}", file=sys.stderr)
        sys.exit(1)

    import yaml

    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # KAUST experiment overrides from config_kaust_data.yaml (paper: lr=0.01, split_method=random, etc.)
    kaust_config_path = base_config_path.parent / "config_kaust_data.yaml"
    if kaust_config_path.exists():
        with open(kaust_config_path, "r", encoding="utf-8") as f:
            kaust_overrides = yaml.safe_load(f)
        base_config.update(kaust_overrides)

    # CLI overrides
    base_config["data_file"] = args.data_file
    base_config["n_experiments"] = args.n_experiments
    base_config["base_seed"] = args.base_seed
    # δ + P_nc(δ) only for DA-STDK (applied per combo below)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base = repo_root / "results" / f"kaust_data_{timestamp}"
    results_base.mkdir(parents=True, exist_ok=True)
    print(f"Results base: {results_base}")
    if args.use_delta_reparameterization:
        print(
            f"δ reparameterization + P_nc(δ): DA-STDK only  |  non_crossing_lambda: {args.non_crossing_lambda}"
        )

    for scenario_name, (obs_method, obs_spatial_pattern) in SCENARIOS.items():
        for model_name, (spatial_learnable, spatial_init_method) in MODELS.items():
            combo_name = f"{scenario_name}_{model_name}"
            config = dict(base_config)
            config["obs_method"] = obs_method
            config["obs_spatial_pattern"] = obs_spatial_pattern
            config["spatial_learnable"] = spatial_learnable
            config["spatial_init_method"] = spatial_init_method
            config["tag"] = f"kaust_{combo_name}"
            # δ reparameterization + P_nc(δ): only for DA-STDK (STDK: no δ)
            if model_name == "DA-STDK" and args.use_delta_reparameterization:
                config["use_delta_reparameterization"] = True
                config["non_crossing_lambda"] = args.non_crossing_lambda
            else:
                config["use_delta_reparameterization"] = False
                config["non_crossing_lambda"] = 0.0

            out_dir = results_base / combo_name
            out_dir.mkdir(parents=True, exist_ok=True)
            config_path = out_dir / "config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)

            cmd = [
                sys.executable,
                str(train_script),
                "--config",
                str(config_path),
                "--output_dir",
                str(out_dir),
                "--n_experiments",
                str(args.n_experiments),
                "--base_seed",
                str(args.base_seed),
            ]
            if args.parallel:
                cmd.append("--parallel")
            print(f"Run: {combo_name}")
            if args.dry_run:
                print(" ", " ".join(cmd))
                continue
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)
            ret = subprocess.run(cmd, cwd=str(repo_root), env=env)
            if ret.returncode != 0:
                print(f"Failed: {combo_name} (exit {ret.returncode})", file=sys.stderr)
                sys.exit(ret.returncode)

    print(f"Done. Results: {results_base}")

    if args.analyze and not args.dry_run and analyze_script.exists():
        print("Running analyzer...")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        analyze_cmd = [sys.executable, str(analyze_script), "--results_dir", str(results_base)]
        if args.skip_coverage_plot:
            analyze_cmd.append("--skip_coverage_plot")
        if args.skip_spatial_compare_plot:
            analyze_cmd.append("--skip_spatial_compare_plot")
        if args.paper_main_only:
            print(
                "Note: --paper_main_only is not used by analyze_kaust_results.py; "
                "call da_stdk.viz.create_paper_main_figures separately if needed.",
                file=sys.stderr,
            )
        ret = subprocess.run(analyze_cmd, cwd=str(repo_root), env=env)
        if ret.returncode != 0:
            print(f"Analyzer exited with {ret.returncode}", file=sys.stderr)
    else:
        print(
            f"Analyze: poetry run python scripts/analyze_kaust_results.py --results_dir {results_base}"
        )


if __name__ == "__main__":
    main()
