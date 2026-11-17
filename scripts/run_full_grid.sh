#!/usr/bin/env bash
# Full factorial sweep of the binary-classification pipeline on the bundled
# example data: normalizer x reducer x selector x feature number x classifier.
# Balance is single-valued in FAE, so it stays at "none".
#
# "none" is excluded from NORMALIZERS by default: raw radiomics features span
# many orders of magnitude and make SVM / Gaussian process / MLP training
# pathologically slow. Include it explicitly if you need that baseline:
#   NORMALIZERS="none zscore minmax mean" bash scripts/run_full_grid.sh
#
# Usage:
#   bash scripts/run_full_grid.sh [output_dir] [feature_numbers]
#
# Examples:
#   bash scripts/run_full_grid.sh                        # 1440 models
#   bash scripts/run_full_grid.sh fae_runs/fast "10"     # 360 models
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${1:-fae_runs/full_grid}"
NUMBERS="${2:-1 5 10 20}"
NORMALIZERS="${NORMALIZERS:-zscore minmax mean}"

# shellcheck disable=SC2086  # the two variables hold space-separated lists
python -m fae bc \
  --data Example/numeric_feature.csv \
  --normalizer $NORMALIZERS \
  --reducer none pcc pca \
  --selector none anova kw rfe \
  --feature-numbers $NUMBERS \
  --classifier svm lr lasso lda rf dt ab nb gp ae \
  --cv 5 --seed 2026 --no-hyper-param \
  --output "$OUT" --top 30
