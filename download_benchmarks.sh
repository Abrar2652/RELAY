#!/bin/bash
# Download MaxSAT Evaluation benchmarks for years 2018-2024.
#
# Creates the following directory structure:
#   benchmarks/
#   ├── train/              # WMS+(2MB) from 2018 — used for pre-training
#   ├── ms2020/             # Unweighted MaxSAT 2020
#   ├── wms2020/            # Weighted MaxSAT 2020
#   ├── ms2021/ ... ms2024/
#   └── wms2021/ ... wms2024/
#
# Paper reference: SGAT-MS uses "non-partial unweighted and weighted benchmark
# instances from MaxSAT Evaluations" — specifically the incomplete track.

set -e

BENCH_DIR="$(dirname "$0")/benchmarks"
DOWNLOAD_DIR="${BENCH_DIR}/_downloads"
mkdir -p "$DOWNLOAD_DIR"

echo "============================================================"
echo "  MaxSAT Evaluation Benchmark Downloader"
echo "============================================================"

# ---- 2018 (training data) ----
echo ""
echo "[2018] Downloading incomplete track benchmarks..."

if [ ! -f "${DOWNLOAD_DIR}/ms18_incomplete_unwt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms18_incomplete_unwt.zip" \
        "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms18_incomplete_unwt.zip" || \
    echo "WARNING: Failed to download 2018 unweighted incomplete"
fi

if [ ! -f "${DOWNLOAD_DIR}/ms18_incomplete_wt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms18_incomplete_wt.zip" \
        "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms18_incomplete_wt.zip" || \
    echo "WARNING: Failed to download 2018 weighted incomplete"
fi

# ---- 2020 (test data) ----
echo ""
echo "[2020] Downloading incomplete track benchmarks..."

if [ ! -f "${DOWNLOAD_DIR}/ms20_incomplete_unwt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms20_incomplete_unwt.zip" \
        "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms20_incomplete_unwt.zip" || \
    echo "WARNING: Failed to download 2020 unweighted incomplete"
fi

if [ ! -f "${DOWNLOAD_DIR}/ms20_incomplete_wt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms20_incomplete_wt.zip" \
        "http://www.cs.toronto.edu/maxsat-lib/maxsat-instances/downloads/ms-evals/ms20_incomplete_wt.zip" || \
    echo "WARNING: Failed to download 2020 weighted incomplete"
fi

# ---- 2021 (test data) ----
echo ""
echo "[2021] Downloading incomplete track benchmarks..."

if [ ! -f "${DOWNLOAD_DIR}/ms21_incomplete_unwt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms21_incomplete_unwt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/mse2021/mse2021_benchmarks/mse21_incomplete_unwt.zip" || \
    echo "WARNING: Failed to download 2021 unweighted incomplete"
fi

if [ ! -f "${DOWNLOAD_DIR}/ms21_incomplete_wt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms21_incomplete_wt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/mse2021/mse2021_benchmarks/mse21_incomplete_wt.zip" || \
    echo "WARNING: Failed to download 2021 weighted incomplete"
fi

# ---- 2022 (test data) ----
echo ""
echo "[2022] Downloading incomplete track benchmarks..."

if [ ! -f "${DOWNLOAD_DIR}/ms22_incomplete_unwt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms22_incomplete_unwt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/MSE2022-inc-instances/mse22-incomplete-unweighted.zip" || \
    echo "WARNING: Failed to download 2022 unweighted incomplete"
fi

if [ ! -f "${DOWNLOAD_DIR}/ms22_incomplete_wt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms22_incomplete_wt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/MSE2022-inc-instances/mse22-incomplete-weighted.zip" || \
    echo "WARNING: Failed to download 2022 weighted incomplete"
fi

# ---- 2023 (test data) ----
echo ""
echo "[2023] Downloading anytime track benchmarks..."

if [ ! -f "${DOWNLOAD_DIR}/ms23_anytime_unwt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms23_anytime_unwt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/MSE2023-anytime-instances/MSE2023-anytime-UW-benchmarks.zip" || \
    echo "WARNING: Failed to download 2023 unweighted anytime"
fi

if [ ! -f "${DOWNLOAD_DIR}/ms23_anytime_wt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms23_anytime_wt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/MSE2023-anytime-instances/MSE2023-anytime-W-benchmarks.zip" || \
    echo "WARNING: Failed to download 2023 weighted anytime"
fi

# ---- 2024 (test data) ----
# 2024 benchmarks may be on a different host; try common locations
echo ""
echo "[2024] Downloading benchmarks..."

if [ ! -f "${DOWNLOAD_DIR}/ms24_unwt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms24_unwt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/MSE2024-instances/MSE2024-UW-benchmarks.zip" 2>/dev/null || \
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms24_unwt.zip" \
        "https://maxsat-evaluations.github.io/2024/benchmarks/MSE2024-UW-benchmarks.zip" 2>/dev/null || \
    echo "WARNING: Failed to download 2024 unweighted — try manually from maxsat-evaluations.github.io/2024/"
fi

if [ ! -f "${DOWNLOAD_DIR}/ms24_wt.zip" ]; then
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms24_wt.zip" \
        "https://www.cs.helsinki.fi/group/coreo/MSE2024-instances/MSE2024-W-benchmarks.zip" 2>/dev/null || \
    wget -q --show-progress -O "${DOWNLOAD_DIR}/ms24_wt.zip" \
        "https://maxsat-evaluations.github.io/2024/benchmarks/MSE2024-W-benchmarks.zip" 2>/dev/null || \
    echo "WARNING: Failed to download 2024 weighted — try manually from maxsat-evaluations.github.io/2024/"
fi

echo ""
echo "============================================================"
echo "  Downloads complete. Now extracting..."
echo "============================================================"

# ---- Extract and organize ----

extract_to() {
    local zip_file="$1"
    local target_dir="$2"
    if [ -f "$zip_file" ] && [ -s "$zip_file" ]; then
        mkdir -p "$target_dir"
        echo "  Extracting $(basename "$zip_file") -> $target_dir"
        unzip -q -o "$zip_file" -d "$target_dir" 2>/dev/null || \
            echo "  WARNING: Failed to extract $zip_file"
    else
        echo "  SKIP: $(basename "$zip_file") not found or empty"
    fi
}

# 2018 → train
extract_to "${DOWNLOAD_DIR}/ms18_incomplete_unwt.zip" "${BENCH_DIR}/ms2018"
extract_to "${DOWNLOAD_DIR}/ms18_incomplete_wt.zip"   "${BENCH_DIR}/wms2018"

# 2020-2024 → test
for year in 2020 2021 2022 2023 2024; do
    case $year in
        2020) uw="ms20_incomplete_unwt.zip"; wt="ms20_incomplete_wt.zip" ;;
        2021) uw="ms21_incomplete_unwt.zip"; wt="ms21_incomplete_wt.zip" ;;
        2022) uw="ms22_incomplete_unwt.zip"; wt="ms22_incomplete_wt.zip" ;;
        2023) uw="ms23_anytime_unwt.zip";    wt="ms23_anytime_wt.zip" ;;
        2024) uw="ms24_unwt.zip";            wt="ms24_wt.zip" ;;
    esac
    extract_to "${DOWNLOAD_DIR}/${uw}" "${BENCH_DIR}/ms${year}"
    extract_to "${DOWNLOAD_DIR}/${wt}" "${BENCH_DIR}/wms${year}"
done

echo ""
echo "============================================================"
echo "  Organizing train split (WMS+ 2018 < 2MB)..."
echo "============================================================"

# Build training split: all 2018 instances < 2MB
TRAIN_DIR="${BENCH_DIR}/train"
mkdir -p "$TRAIN_DIR"

for src_dir in "${BENCH_DIR}/ms2018" "${BENCH_DIR}/wms2018"; do
    if [ -d "$src_dir" ]; then
        find "$src_dir" -type f \( -name "*.cnf" -o -name "*.wcnf" \) -size -2M | while read f; do
            cp "$f" "$TRAIN_DIR/"
        done
    fi
done

TRAIN_COUNT=$(find "$TRAIN_DIR" -type f \( -name "*.cnf" -o -name "*.wcnf" \) 2>/dev/null | wc -l)
echo "  Training instances (< 2MB from 2018): $TRAIN_COUNT"

echo ""
echo "============================================================"
echo "  Final benchmark statistics:"
echo "============================================================"

for d in train ms2018 wms2018 ms2020 wms2020 ms2021 wms2021 ms2022 wms2022 ms2023 wms2023 ms2024 wms2024; do
    dir="${BENCH_DIR}/${d}"
    if [ -d "$dir" ]; then
        count=$(find "$dir" -type f \( -name "*.cnf" -o -name "*.wcnf" \) 2>/dev/null | wc -l)
        printf "  %-15s %5d instances\n" "$d" "$count"
    else
        printf "  %-15s %5s\n" "$d" "MISSING"
    fi
done

echo ""
echo "Done! Benchmarks are in: $BENCH_DIR"
