#!/usr/bin/env python3
"""
Verify a single experiment run against reference numbers from a KAUST experiment run (e.g. table_4_4_20260203_153043 or kaust_data_<timestamp>).

Reference metrics (Fixed_Uniform_STDK, experiment 1, seed 2025):
  test_crps: 0.1895, test_coverage_90: 0.919, test_mse: 0.1116, valid_crps: 0.1914, train_crps: 0.1284

Usage:
  # 1) Run one experiment (~18 min):
  PYTHONPATH=. python scripts/train_default.py \\
    --config configs/config_verify_single_fixed_uniform_stdk.yaml \\
    --output_dir results/verify_single/Fixed_Uniform_STDK --n_experiments 1 --base_seed 2025
  # 2) Compare to reference:
  python scripts/verify_single_experiment.py --results_dir results/verify_single/Fixed_Uniform_STDK
  # Print reference values only:
  python scripts/verify_single_experiment.py --reference_only
"""
import argparse
import json
from pathlib import Path

# Reference: KAUST experiment run table_4_4_20260203_153043, Fixed_Uniform_STDK experiment 1 (seed 2025)
REFERENCE_DIR = (
    Path(__file__).resolve().parents[1] / "table_4_4_20260203_153043" / "Fixed_Uniform_STDK"
)
REFERENCE_EXP_ID = 1

# Keys to compare (float); relative tolerance for agreement
KEYS_TO_COMPARE = [
    "test_crps",
    "test_coverage_90",
    "test_coverage_90_conformal",
    "test_mse",
    "valid_crps",
    "train_crps",
]
REL_TOL = 0.02  # allow 2% relative difference (e.g. different machine/float)
ABS_TOL = 0.01  # or small absolute difference for near-zero


def load_result(exp_dir: Path, experiment_id: int) -> dict:
    p = exp_dir / "experiments" / str(experiment_id) / "results.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="Verify single experiment vs reference KAUST experiment run"
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Path to run dir (e.g. results/verify_single/Fixed_Uniform_STDK or .../Fixed_Uniform_STDK)",
    )
    ap.add_argument("--reference_only", action="store_true", help="Only print reference values")
    ap.add_argument(
        "--reference_dir",
        type=str,
        default=str(REFERENCE_DIR),
        help="Path to reference run (default: KAUST experiment dir Fixed_Uniform_STDK)",
    )
    ap.add_argument("--exp_id", type=int, default=1, help="Experiment ID to compare (default 1)")
    args = ap.parse_args()

    ref_dir = Path(args.reference_dir)
    ref = load_result(ref_dir, REFERENCE_EXP_ID if not args.reference_dir else args.exp_id)

    print("Reference (KAUST experiment run Fixed_Uniform_STDK experiment 1, seed 2025):")
    print("-" * 50)
    for k in KEYS_TO_COMPARE:
        v = ref.get(k)
        if v is not None:
            print(f"  {k}: {v}")
    print(f"  experiment_seed: {ref.get('experiment_seed')}")
    print()

    if args.reference_only:
        return

    results_dir = args.results_dir
    if not results_dir:
        print("Pass --results_dir <path> to compare your run against the reference.")
        return
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Not found: {results_dir}")
        return

    try:
        new = load_result(results_dir, args.exp_id)
    except FileNotFoundError as e:
        print(f"Your run: {e}")
        return

    print("Your run vs reference:")
    print("-" * 50)
    all_ok = True
    for k in KEYS_TO_COMPARE:
        v_ref = ref.get(k)
        v_new = new.get(k)
        if v_ref is None and v_new is None:
            continue
        if v_ref is None or v_new is None:
            print(f"  {k}: REF={v_ref} NEW={v_new} (missing)")
            all_ok = False
            continue
        try:
            diff = abs(v_new - v_ref)
            reld = diff / (abs(v_ref) + 1e-12)
            ok = diff <= ABS_TOL or reld <= REL_TOL
            status = "OK" if ok else "DIFF"
            if not ok:
                all_ok = False
            print(f"  {k}: ref={v_ref:.6f} new={v_new:.6f} -> {status}")
        except Exception as e:
            print(f"  {k}: error {e}")
            all_ok = False
    print()
    print("Verification:", "PASS" if all_ok else "FAIL (check differences)")


if __name__ == "__main__":
    main()
