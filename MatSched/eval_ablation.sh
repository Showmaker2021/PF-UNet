#!/usr/bin/env bash
# Evaluate all 8 ablation checkpoints across ID, OOD, and Brandimarte Benchmark
MODELS=(
  "abl_full_10x5"
  "abl_no_cc_10x5"
  "abl_no_m2o_10x5"
  "abl_no_midout_10x5"
  "abl_pure_l3_10x5"
  "abl_kd_start1_10x5"
  "abl_layer1_10x5"
  "abl_layer3_10x5"
)

echo "=========================================================="
echo "Evaluating All Ablation Models on Level 3"
echo "=========================================================="

for m in "${MODELS[@]}"; do
  ckpt="MatSched/checkpoints/${m}/best.pt"
  if [ -f "$ckpt" ]; then
    echo ">>> Evaluating $m ..."
    python MatSched/evaluate.py --mode matsched --level 3 --batch_size 200 \
      --ckpt "$ckpt" \
      --datasets data/FJSP/TNNLS/SD1/10x5 data/FJSP/TNNLS/SD1/30x10 data/FJSP/TNNLS/SD1/40x10 data/FJSP/TNNLS/BenchData/Brandimarte
    echo ""
  else
    echo "Checkpoint $ckpt not found, skipping..."
  fi
done
