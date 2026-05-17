#!/usr/bin/env bash
#
# Run on the machine that HAS the experiment results (or run full pipeline).
# Produces KAUST experiment summary and spatial coverage figures (Fig 3/4), then
# copies the 4 spatial PNGs to a folder you can bring back.
#
# Usage:
#   # Option A: You already have a results dir with experiments/.../results.json
#   ./scripts/regenerate_paper_figures.sh results/kaust_data_20260228_123456
#
#   # Option B: Run full pipeline (train + analyze), then copy figures
#   ./scripts/regenerate_paper_figures.sh --run-full [--n 1]
#
#   # Option C: Copy figures to a specific folder (default: paper_figures/)
#   ./scripts/regenerate_paper_figures.sh results/kaust_data_xxx --out-dir .local/fig
#
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"

OUT_DIR="${OUT_DIR:-paper_figures}"
RUN_FULL=""
N_EXP=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --run-full) RUN_FULL=1; shift ;;
    --n)        N_EXP="$2"; shift 2 ;;
    --out-dir)  OUT_DIR="$2"; shift 2 ;;
    *)          break ;;
  esac
done

if [[ -n "$RUN_FULL" ]]; then
  echo "=== Running full pipeline (run_kaust_data.py then analyze) ==="
  N="${N_EXP:-1}"
  python scripts/run_kaust_data.py --n_experiments "$N" --analyze
  # run_kaust_data.py writes to results/kaust_data_<timestamp>; find latest
  RESULTS_DIR=$(ls -td results/kaust_data_* 2>/dev/null | head -1)
  if [[ -z "$RESULTS_DIR" ]]; then
    echo "Error: No results/kaust_data_* directory found."
    exit 1
  fi
  echo "Using results: $RESULTS_DIR"
else
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <results_dir> [--out-dir DIR]"
    echo "   or: $0 --run-full [--n 1] [--out-dir DIR]"
    echo ""
    echo "Examples:"
    echo "  $0 results/kaust_data_20260228_123456"
    echo "  $0 --run-full --n 1"
    exit 1
  fi
  RESULTS_DIR="$1"
  shift
  if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Error: Results directory not found: $RESULTS_DIR"
    exit 1
  fi
  echo "=== Analyzing existing results: $RESULTS_DIR ==="
  python scripts/analyze_kaust_results.py --results_dir "$RESULTS_DIR"
fi

SUMMARY="$RESULTS_DIR/summary"
mkdir -p "$OUT_DIR"

for name in spatial_coverage_compare_fixed_uniform spatial_coverage_compare_fixed_clustered spatial_coverage_compare_random_uniform spatial_coverage_compare_random_clustered; do
  SRC="$SUMMARY/${name}.png"
  if [[ -f "$SRC" ]]; then
    cp "$SRC" "$OUT_DIR/"
    echo "Copied: $OUT_DIR/${name}.png"
  else
    echo "Skip (not found): $SRC"
  fi
done

echo ""
echo "Done. Figures are in: $OUT_DIR"
