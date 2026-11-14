#!/usr/bin/env bash
set -e

# Script to run comprehensive ablation study on SD1 10x5 (400 iters)
echo "=========================================================="
echo "Starting MatSched Ablation Suite on SD1 10x5"
echo "=========================================================="

mkdir -p MatSched/checkpoints

# 1. Full MatSched-Net (Reference model)
echo "[1/8] Running Full MatSched-Net..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 2 \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 100 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_full_10x5

# 2. w/o Core-Context Co-Attention (no_cc)
echo "[2/8] Running Ablation: w/o Co-Attention (no_cc)..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 2 \
  --no_cc \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 100 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_no_cc_10x5

# 3. w/o Machine-to-Operation feedback (no_m2o)
echo "[3/8] Running Ablation: w/o M2O feedback (no_m2o)..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 2 \
  --no_m2o \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 100 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_no_m2o_10x5

# 4. w/o Mid-ring to output (no_mid_out)
echo "[4/8] Running Ablation: w/o Mid-out (no_mid_out)..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 2 \
  --no_mid_out \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 100 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_no_midout_10x5

# 5. Pure L3 (No Matryoshka / No KD)
echo "[5/8] Running Ablation: Pure L3 (No Matryoshka / No KD)..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 2 \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_pure_l3_10x5

# 6. KD Start from iter 1 (Immediate Distillation without warm-up)
echo "[6/8] Running Ablation: KD Start Iter 1..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 2 \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 1 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_kd_start1_10x5

# 7. Layer = 1 (Shallow transformer)
echo "[7/8] Running Ablation: Layer = 1..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 1 \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 100 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_layer1_10x5

# 8. Layer = 3 (Deeper transformer)
echo "[8/8] Running Ablation: Layer = 3..."
python MatSched/train_ppo.py \
  --dataset SD1 --num_jobs 10 --num_machines 5 --num_layers 3 \
  --iters 400 --episodes_per_iter 50 --batch_size 50 --minibatch_size 1024 \
  --enable_kd --kd_start_iter 100 \
  --val_path data/FJSP/TNNLS/data_train_vali/SD1/10x5 \
  --save_dir MatSched/checkpoints/abl_layer3_10x5

echo "=========================================================="
echo "All 8 Ablation runs completed successfully!"
echo "=========================================================="
