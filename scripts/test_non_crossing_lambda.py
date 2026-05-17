#!/usr/bin/env python3
"""
Quick test: compare non_crossing_lambda=0 vs 1.0 (1 experiment each, short epochs).
Checks if P_nc(δ) penalty has an effect on test CRPS / crossing behavior.
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "config_default.yaml"
    train_script = repo_root / "scripts" / "train_default.py"
    out_base = repo_root / "results" / "lambda_test"
    out_base.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    # Short run: 1 experiment, 30 epochs
    base["use_delta_reparameterization"] = True
    base["n_experiments"] = 1
    base["epochs"] = 30
    base["base_seed"] = 42
    base["data_file"] = base.get("data_file", "data/2a/2a_8.csv")

    results = {}
    for lam, label in [(0.0, "lambda_0"), (1.0, "lambda_1")]:
        base["non_crossing_lambda"] = lam
        base["tag"] = f"lambda_test_{label}"
        out_dir = out_base / label
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(base, f, default_flow_style=False)

        env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}
        ret = subprocess.run(
            [
                sys.executable,
                str(train_script),
                "--config",
                str(cfg_path),
                "--output_dir",
                str(out_dir),
                "--n_experiments",
                "1",
                "--base_seed",
                "42",
            ],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if ret.returncode != 0:
            print(f"[{label}] run failed: {ret.stderr[:500]}")
            results[label] = {"test_crps": None, "ok": False}
            continue

        res_file = out_dir / "experiments" / "1" / "results.json"
        if not res_file.exists():
            results[label] = {"test_crps": None, "ok": False}
            continue
        with open(res_file, "r") as f:
            data = json.load(f)
        results[label] = {
            "test_crps": data.get("test_crps"),
            "train_crps": data.get("train_crps"),
            "ok": True,
        }

    print("\n" + "=" * 60)
    print("non_crossing_lambda quick test (1 run each, 30 epochs)")
    print("=" * 60)
    for label in ["lambda_0", "lambda_1"]:
        r = results.get(label, {})
        if r.get("ok"):
            print(f"  {label}:  test_crps={r['test_crps']:.6f}  train_crps={r['train_crps']:.6f}")
        else:
            print(f"  {label}:  (run failed or no results)")
    if results.get("lambda_0", {}).get("ok") and results.get("lambda_1", {}).get("ok"):
        t0 = results["lambda_0"]["test_crps"]
        t1 = results["lambda_1"]["test_crps"]
        print(f"  delta test_crps (lambda_1 - lambda_0): {t1 - t0:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
