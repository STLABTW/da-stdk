#!/usr/bin/env python3
"""
Grid search over non_crossing_lambda (P_nc(δ) penalty weight).

Runs train_default for each λ with use_delta_reparameterization=True,
collects test_crps (mean ± std over replicates), saves CSV and prints table.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def main():
    ap = argparse.ArgumentParser(description="Grid search over non_crossing_lambda")
    ap.add_argument(
        "--config", type=str, default="configs/config_default.yaml", help="Base config YAML"
    )
    ap.add_argument(
        "--lambdas",
        type=str,
        default=None,
        help="Comma-separated λ values; if not set, use 0~0.1 in 10 steps (0,0.01,...,0.1)",
    )
    ap.add_argument("--n_experiments", type=int, default=3, help="Replicates per λ")
    ap.add_argument("--epochs", type=int, default=50, help="Epochs per run")
    ap.add_argument(
        "--data_file", type=str, default=None, help="Override data file (default: from config)"
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output base dir (default: results/lambda_grid_<timestamp>)",
    )
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="Run λ configs in parallel (each config runs n_experiments sequentially)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / args.config
    train_script = repo_root / "scripts" / "train_default.py"
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    if not train_script.exists():
        print(f"Train script not found: {train_script}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    if args.lambdas is None:
        # [0, 0.1] in log scale: 0 + 10 points from 1e-3 to 0.1
        lambdas = [0.0] + np.logspace(-3, -1, 10).tolist()  # 0, 0.001, ..., 0.1 (log-spaced)
    else:
        lambdas = [float(x.strip()) for x in args.lambdas.split(",")]
    base["use_delta_reparameterization"] = True
    base["non_crossing_l1_penalization"] = True  # ensure Eq. (8) ℓ₁ penalization is enabled
    base["n_experiments"] = args.n_experiments
    base["epochs"] = args.epochs
    base["base_seed"] = 2025
    if args.data_file:
        base["data_file"] = args.data_file

    if args.output_dir:
        out_base = Path(args.output_dir)
    else:
        out_base = repo_root / "results" / f"lambda_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_base.mkdir(parents=True, exist_ok=True)
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}

    def run_one(lam):
        label = f"lambda_{lam:.2f}".replace(".", "_")
        out_dir = out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg = dict(base)
        cfg["non_crossing_lambda"] = lam
        cfg["tag"] = f"lambda_grid_{label}"
        cfg_path = out_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        ret = subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--config",
                str(cfg_path),
                "--output_dir",
                str(out_dir),
                "--n_experiments",
                str(args.n_experiments),
                "--base_seed",
                "2025",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if ret.returncode != 0:
            return lam, None, None, False
        crps_list = []
        for i in range(1, args.n_experiments + 1):
            res_file = out_dir / "experiments" / str(i) / "results.json"
            if not res_file.exists():
                continue
            with open(res_file, "r") as f:
                data = json.load(f)
            if "test_crps" in data:
                crps_list.append(data["test_crps"])
        if not crps_list:
            return lam, None, None, False
        return lam, float(np.mean(crps_list)), float(np.std(crps_list)), True

    if args.parallel:
        from joblib import Parallel, delayed

        rows = Parallel(n_jobs=min(len(lambdas), 4), verbose=10)(
            delayed(run_one)(lam) for lam in lambdas
        )
    else:
        rows = []
        for i, lam in enumerate(lambdas):
            print(f"[{i+1}/{len(lambdas)}] non_crossing_lambda={lam}")
            rows.append(run_one(lam))

    # Summary table
    summary = []
    for lam, mean_crps, std_crps, ok in rows:
        summary.append(
            {
                "non_crossing_lambda": lam,
                "test_crps_mean": mean_crps if ok else None,
                "test_crps_std": std_crps if ok else None,
                "n_experiments": args.n_experiments if ok else 0,
                "ok": ok,
            }
        )

    # Print
    print("\n" + "=" * 70)
    print("Grid search: non_crossing_lambda (use_delta_reparameterization=True)")
    print(f"  n_experiments={args.n_experiments}, epochs={args.epochs}")
    print("=" * 70)
    print(f"{'lambda':>10}  {'test_crps_mean':>14}  {'test_crps_std':>12}  {'status':>8}")
    print("-" * 70)
    for s in summary:
        if s["ok"]:
            print(
                f"{s['non_crossing_lambda']:>10.2f}  {s['test_crps_mean']:>14.6f}  {s['test_crps_std']:>12.6f}  {'ok':>8}"
            )
        else:
            print(f"{s['non_crossing_lambda']:>10.2f}  {'—':>14}  {'—':>12}  {'fail':>8}")
    print("=" * 70)

    # Save CSV
    import csv

    csv_path = out_base / "grid_search_lambda_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "non_crossing_lambda",
                "test_crps_mean",
                "test_crps_std",
                "n_experiments",
                "ok",
            ],
        )
        w.writeheader()
        for s in summary:
            w.writerow({k: s[k] for k in w.fieldnames})
    print(f"Summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
