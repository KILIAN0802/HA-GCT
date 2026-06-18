#!/bin/bash
set -euo pipefail

# Print GPU information
GPU_ID=${CUDA_VISIBLE_DEVICES:-"not set"}
echo "========================================================"
echo "Starting MultiVSL200 Training & Ensemble Pipeline"
echo "Active GPU: $GPU_ID"
echo "Pipeline start time: $(date)"
echo "========================================================"

# Create a unique run ID for this ensemble pipeline to keep checkpoints and logs organized
ENSEMBLE_RUN_ID="ensemble_$(date +%Y%m%d_%H%M%S)"
echo "Ensemble Run ID: $ENSEMBLE_RUN_ID"

# Create log and checkpoint directories
mkdir -p "logs/$ENSEMBLE_RUN_ID"
mkdir -p "checkpoints/$ENSEMBLE_RUN_ID"

SEEDS=(42 43 44)
MODEL_PATHS=()

# 1. Train 3 models sequentially
for SEED in "${SEEDS[@]}"; do
    echo "--------------------------------------------------------"
    echo "[STAGE 1/2] Training Model with Seed: $SEED"
    echo "Start Time: $(date)"
    echo "Logs will be written to: logs/${ENSEMBLE_RUN_ID}/seed_${SEED}.log"
    echo "--------------------------------------------------------"

    python train.py \
        --dataset multivsl200 \
        --data-dir "/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy" \
        --pretrain-path "" \
        --seed "$SEED" \
        --model-type earlyfusion \
        --d-model 256 \
        --epochs 500 \
        --batch-size 16 \
        --accum-steps 8 \
        --lr 7e-4 \
        --mixup-alpha 0.05 \
        --warmup-epochs 5 \
        --drop-path-max 0.05 \
        --label-smoothing 0.05 \
        --save-dir "checkpoints/${ENSEMBLE_RUN_ID}/seed_${SEED}/" \
        > "logs/${ENSEMBLE_RUN_ID}/seed_${SEED}.log" 2>&1

    echo "Finished training for Seed $SEED at $(date)"

    # 2. Capture best checkpoint path automatically
    BEST_PATH=$(find "checkpoints/${ENSEMBLE_RUN_ID}/seed_${SEED}/" -name "best_ha_gct_model.pth" | sort | tail -n 1)
    if [ -z "$BEST_PATH" ] || [ ! -f "$BEST_PATH" ]; then
        echo "ERROR: Best checkpoint not found for Seed $SEED!"
        exit 1
    fi
    echo "Captured Best Checkpoint: $BEST_PATH"
    MODEL_PATHS+=("$BEST_PATH")
done

# Store checkpoint paths in environment variables
export MODEL1_PATH="${MODEL_PATHS[0]}"
export MODEL2_PATH="${MODEL_PATHS[1]}"
export MODEL3_PATH="${MODEL_PATHS[2]}"

# 3. Ensemble Evaluation
echo "========================================================"
echo "[STAGE 2/2] Running Ensemble Evaluation"
echo "Start Time: $(date)"
echo "Model 1 Path: $MODEL1_PATH"
echo "Model 2 Path: $MODEL2_PATH"
echo "Model 3 Path: $MODEL3_PATH"
echo "========================================================"

python -c '
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Extend path to allow imports from workspace root
sys.path.extend(["./", "../"])

from utils.preprocessing import SkeletonTransforms
from data.dataloader import get_multivsl_loaders
from models.ha_gct import EarlyFusionHA_GCT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ensemble Evaluation Device: {device}")

model_paths = [os.environ.get("MODEL1_PATH"), os.environ.get("MODEL2_PATH"), os.environ.get("MODEL3_PATH")]

# Load all 3 models
models = []
for idx, path in enumerate(model_paths):
    if not path or not os.path.exists(path):
        raise ValueError(f"Model path {path} does not exist!")
    
    model = EarlyFusionHA_GCT(
        num_joints=27,
        in_channels=2,
        d_model=256,
        num_ha_gc_blocks=3,
        num_mhsa_layers=2,
        nhead=4,
        num_classes=199,
        dropout=0.1,
        graph_lambda=0.05,
        max_frames=150,
        drop_path_max=0.05
    ).to(device)
    
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    models.append(model)
    print(f"Loaded Model {idx+1} from {path}")

# Load test dataset
transform = SkeletonTransforms(
    num_joints=27,
    max_frames=150,
    verbose=False
)

_, _, test_loader = get_multivsl_loaders(
    "/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy",
    batch_size=32,
    num_workers=4,
    transform=transform,
    split_method="random"
)

correct_top1 = 0
correct_top5 = 0
total = 0

with torch.no_grad():
    for batch_data, batch_labels in test_loader:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        # Average logits directly: (logits1 + logits2 + logits3) / 3
        logits1 = models[0](batch_data)
        logits2 = models[1](batch_data)
        logits3 = models[2](batch_data)
        
        logits = (logits1 + logits2 + logits3) / 3.0
        
        num_classes = logits.size(1)
        maxk = min(5, num_classes)
        
        # Calculate Top-1 and Top-5 accuracy
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(batch_labels.view(-1, 1).expand_as(pred))
        
        correct_top1 += correct[:, :1].reshape(-1).float().sum().item()
        correct_top5 += correct[:, :maxk].reshape(-1).float().sum().item()
        total += batch_labels.size(0)

top1_acc = 100.0 * correct_top1 / total
top5_acc = 100.0 * correct_top5 / total

print("\n" + "=" * 50)
print("FINAL ENSEMBLE RESULTS")
print("=" * 50)
print(f"Total test samples: {total}")
print(f"Top-1 Accuracy:     {top1_acc:.2f}%")
print(f"Top-5 Accuracy:     {top5_acc:.2f}%")
print("=" * 50)
'

echo "========================================================"
echo "Pipeline finished successfully at $(date)"
echo "========================================================"
