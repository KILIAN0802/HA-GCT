#!/bin/bash
set -euo pipefail

# ============================================================================
# HA-GCT 4-Stream Multi-Stream Fast Training Script
# Architecture: Joint + Bone + Motion + BoneMotion with Learnable Weighted Fusion
# Model: d_model=128, Graph-Augmented Attention with skeleton topology bias
# ============================================================================

GPU_ID=${CUDA_VISIBLE_DEVICES:-"not set"}
echo "========================================================"
echo "HA-GCT 4-Stream Multi-Stream Fast Training"
echo "Active GPU: $GPU_ID"
echo "Pipeline start time: $(date)"
echo "========================================================"

# Generate unique run ID
RUN_ID="multistream_4s_$(date +'%Y%m%d_%H%M%S')"
echo "Run ID: $RUN_ID"

# Create directories
mkdir -p "logs/$RUN_ID"
mkdir -p "checkpoints/$RUN_ID"

# Python executable
PYTHON_EXEC="/mnt/nvme0/home/utbt_sv1/miniconda3/envs/haslr_env/bin/python"

# ============================================================================
# CONFIGURATION (optimized for 12GB+ VRAM with d_model=128, 4 streams)
# ============================================================================
DATASET="multivsl200"
DATA_DIR="/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy"
D_MODEL=128
NUM_CLASSES=199
BATCH_SIZE=16          # Safe for 12GB VRAM with 4 streams x d_model=128
ACCUM_STEPS=4          # Effective batch = 16 * 4 = 64
LR=3e-4
EPOCHS=500
PATIENCE=50
DROPOUT=0.3
DROP_PATH=0.2
LABEL_SMOOTHING=0.1
MIXUP_ALPHA=0.2
WARMUP_EPOCHS=10
SEED=42
LOSS_FN="focal"

echo "Configuration:"
echo "  Dataset:          $DATASET ($NUM_CLASSES classes)"
echo "  d_model:          $D_MODEL"
echo "  Batch size:       $BATCH_SIZE (effective: $((BATCH_SIZE * ACCUM_STEPS)))"
echo "  Learning rate:    $LR"
echo "  Epochs:           $EPOCHS (patience: $PATIENCE)"
echo "  Dropout:          $DROPOUT"
echo "  DropPath:         $DROP_PATH"
echo "  Label Smoothing:  $LABEL_SMOOTHING"
echo "  Mixup Alpha:      $MIXUP_ALPHA"
echo "  Loss Function:    $LOSS_FN"
echo "  Seed:             $SEED"
echo ""

# ============================================================================
# STAGE 1: Self-Supervised Pre-Training (Optional but recommended)
# ============================================================================
echo "--------------------------------------------------------"
echo "[STAGE 1/2] Self-Supervised Pre-Training"
echo "Start Time: $(date)"
echo "Logs: logs/${RUN_ID}/pretrain.log"
echo "--------------------------------------------------------"

$PYTHON_EXEC train.py \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --pretrain \
    --pretrain-epochs 50 \
    --pretrain-path "checkpoints/${RUN_ID}/pretrained_ha_gct.pth" \
    --seed "$SEED" \
    --d-model "$D_MODEL" \
    --batch-size "$BATCH_SIZE" \
    --accum-steps "$ACCUM_STEPS" \
    --lr 5e-4 \
    --save-dir "checkpoints/${RUN_ID}/" \
    --use-wandb \
    > "logs/${RUN_ID}/pretrain.log" 2>&1

echo "Pre-training completed at $(date)"

# ============================================================================
# STAGE 2: Fine-Tuning with 4-Stream Multi-Stream Architecture
# ============================================================================
echo "--------------------------------------------------------"
echo "[STAGE 2/2] Fine-Tuning 4-Stream Multi-Stream HA-GCT"
echo "Start Time: $(date)"
echo "Logs: logs/${RUN_ID}/finetune.log"
echo "--------------------------------------------------------"

$PYTHON_EXEC train.py \
    --dataset "$DATASET" \
    --data-dir "$DATA_DIR" \
    --pretrain-path "checkpoints/${RUN_ID}/pretrained_ha_gct.pth" \
    --seed "$SEED" \
    --model-type multistream \
    --d-model "$D_MODEL" \
    --num-classes "$NUM_CLASSES" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch-size "$BATCH_SIZE" \
    --accum-steps "$ACCUM_STEPS" \
    --lr "$LR" \
    --classifier-lr-mult 3.0 \
    --mixup-alpha "$MIXUP_ALPHA" \
    --warmup-epochs "$WARMUP_EPOCHS" \
    --drop-path-max "$DROP_PATH" \
    --dropout "$DROPOUT" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --crop-min-ratio 0.5 \
    --class-balanced \
    --loss-fn "$LOSS_FN" \
    --save-dir "checkpoints/${RUN_ID}/" \
    --use-wandb \
    > "logs/${RUN_ID}/finetune.log" 2>&1

echo "Fine-tuning completed at $(date)"

# ============================================================================
# RESULTS
# ============================================================================
echo "========================================================"
echo "Training pipeline completed at $(date)"
echo "Run ID: $RUN_ID"
echo "Checkpoint dir: checkpoints/${RUN_ID}/"
echo "Log dir: logs/${RUN_ID}/"
echo ""
echo "To view training logs:"
echo "  tail -f logs/${RUN_ID}/finetune.log"
echo ""
echo "To resume training if interrupted:"
echo "  python train.py --resume checkpoints/${RUN_ID}/best_ha_gct_model.pth ..."
echo "========================================================"
