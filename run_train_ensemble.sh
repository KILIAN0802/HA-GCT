#!/bin/bash

# ==============================================================================
# HA-GCT ENSEMBLE TRAINING SCRIPT (PHASE 4)
# Runs training for 5 independent models sequentially.
# ==============================================================================

# 1. Configuration Variables (Edit these if needed)
GPU_ID=3
DATA_DIR="/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy"
DATASET="multivsl200"
EPOCHS=200
BATCH_SIZE=32
LR="3e-4"
WANDB_PROJECT="HA-GCT"

# Pre-training configurations
PRETRAIN_EPOCHS=50
PRETRAIN_PATH="checkpoints/pretrained_ha_gct.pth"

# Export CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Create log directory if not exists
mkdir -p results/logs

echo "=============================================================================="
echo "STARTING SEQUENTIAL ENSEMBLE TRAINING ON GPU $GPU_ID"
echo "Start time: $(date)"
echo "=============================================================================="

# ------------------------------------------------------------------------------
# STAGE 1: SELF-SUPERVISED PRE-TRAINING (MASKED SKELETON AUTOENCODER)
# Runs once to generate the pre-trained encoder weights if not already present.
# ------------------------------------------------------------------------------
if [ ! -f "$PRETRAIN_PATH" ]; then
    echo "Pre-trained weights not found at $PRETRAIN_PATH."
    echo "Starting Stage 1: Self-Supervised Pre-training (MSA) for $PRETRAIN_EPOCHS epochs..."
    START_PRE=$(date +%s)
    python train.py \
        --dataset $DATASET \
        --data-dir $DATA_DIR \
        --pretrain \
        --pretrain-epochs $PRETRAIN_EPOCHS \
        --pretrain-path $PRETRAIN_PATH \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --wandb-project $WANDB_PROJECT \
        --use-wandb
    END_PRE=$(date +%s)
    echo "Stage 1 Pre-training completed. Duration: $((END_PRE - START_PRE))s"
else
    echo "Pre-trained weights found at $PRETRAIN_PATH. Skipping Stage 1 Pre-training."
fi

# ==============================================================================
# STAGE 2: FINE-TUNING CLASSIFICATION (5 MODELS)
# ==============================================================================

# ------------------------------------------------------------------------------
# MODEL 1: seed=42, augmentation set A (crop_min=0.6), multistream
# ------------------------------------------------------------------------------
echo ""
echo "[1/5] Training Model 1 (seed=42, crop_min=0.6, multistream)..."
START_M1=$(date +%s)
python train.py \
    --dataset $DATASET \
    --data-dir $DATA_DIR \
    --pretrain-path $PRETRAIN_PATH \
    --seed 42 \
    --crop-min-ratio 0.6 \
    --model-type multistream \
    --d-model 128 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wandb-project $WANDB_PROJECT \
    --use-wandb
END_M1=$(date +%s)
echo "Model 1 completed. Duration: $((END_M1 - START_M1))s"

# ------------------------------------------------------------------------------
# MODEL 2: seed=123, augmentation set A (crop_min=0.6), multistream
# ------------------------------------------------------------------------------
echo ""
echo "[2/5] Training Model 2 (seed=123, crop_min=0.6, multistream)..."
START_M2=$(date +%s)
python train.py \
    --dataset $DATASET \
    --data-dir $DATA_DIR \
    --pretrain-path $PRETRAIN_PATH \
    --seed 123 \
    --crop-min-ratio 0.6 \
    --model-type multistream \
    --d-model 128 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wandb-project $WANDB_PROJECT \
    --use-wandb
END_M2=$(date +%s)
echo "Model 2 completed. Duration: $((END_M2 - START_M2))s"

# ------------------------------------------------------------------------------
# MODEL 3: seed=42, augmentation set B (crop_min=0.5), multistream
# ------------------------------------------------------------------------------
echo ""
echo "[3/5] Training Model 3 (seed=42, crop_min=0.5, multistream)..."
START_M3=$(date +%s)
python train.py \
    --dataset $DATASET \
    --data-dir $DATA_DIR \
    --pretrain-path $PRETRAIN_PATH \
    --seed 42 \
    --crop-min-ratio 0.5 \
    --model-type multistream \
    --d-model 128 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wandb-project $WANDB_PROJECT \
    --use-wandb
END_M3=$(date +%s)
echo "Model 3 completed. Duration: $((END_M3 - START_M3))s"

# ------------------------------------------------------------------------------
# MODEL 4: seed=42, augmentation set A (crop_min=0.6), earlyfusion (train from scratch)
# ------------------------------------------------------------------------------
echo ""
echo "[4/5] Training Model 4 (seed=42, crop_min=0.6, earlyfusion)..."
START_M4=$(date +%s)
python train.py \
    --dataset $DATASET \
    --data-dir $DATA_DIR \
    --seed 42 \
    --crop-min-ratio 0.6 \
    --model-type earlyfusion \
    --d-model 128 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wandb-project $WANDB_PROJECT \
    --use-wandb
END_M4=$(date +%s)
echo "Model 4 completed. Duration: $((END_M4 - START_M4))s"

# ------------------------------------------------------------------------------
# MODEL 5: seed=42, augmentation set A (crop_min=0.6), multistream, d_model=256
# ------------------------------------------------------------------------------
echo ""
echo "[5/5] Training Model 5 (seed=42, crop_min=0.6, multistream, d_model=256)..."
START_M5=$(date +%s)
python train.py \
    --dataset $DATASET \
    --data-dir $DATA_DIR \
    --pretrain-path $PRETRAIN_PATH \
    --seed 42 \
    --crop-min-ratio 0.6 \
    --model-type multistream \
    --d-model 256 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --wandb-project $WANDB_PROJECT \
    --use-wandb
END_M5=$(date +%s)
echo "Model 5 completed. Duration: $((END_M5 - START_M5))s"

echo ""
echo "=============================================================================="
echo "ALL 5 ENSEMBLE MODELS SUCCESSFULLY TRAINED!"
echo "End time: $(date)"
echo "=============================================================================="
