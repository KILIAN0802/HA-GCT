#!/bin/bash
set -euo pipefail

GPU_ID=${CUDA_VISIBLE_DEVICES:-"not set"}
echo "========================================================"
echo "RESUMING OPTIMIZED MultiVSL200 Training & Ensemble Pipeline"
echo "Active GPU: $GPU_ID"
echo "Pipeline resume time: $(date)"
echo "========================================================"

# Sử dụng lại Run ID cũ
ENSEMBLE_RUN_ID="ensemble_opt_20260621_070453"
echo "Ensemble Run ID: $ENSEMBLE_RUN_ID"

PYTHON_EXEC="/mnt/nvme0/home/utbt_sv1/miniconda3/envs/haslr_env/bin/python"
DATA_DIR="/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy"

MODEL_PATHS=()

# 1. Thêm checkpoint đã hoàn thành của Seed 42
BEST_42="checkpoints/ensemble_opt_20260621_070453/seed_42/20260621_095252/best_ha_gct_model.pth"
if [ ! -f "$BEST_42" ]; then
    echo "ERROR: Seed 42 best model not found at $BEST_42!"
    exit 1
fi
echo "Loaded completed Seed 42 checkpoint: $BEST_42"
MODEL_PATHS+=("$BEST_42")

# 2. Thêm checkpoint đã hoàn thành của Seed 43
BEST_43="checkpoints/ensemble_opt_20260621_070453/seed_43/20260622_111107/best_ha_gct_model.pth"
if [ ! -f "$BEST_43" ]; then
    echo "ERROR: Seed 43 best model not found at $BEST_43!"
    exit 1
fi
echo "Loaded completed Seed 43 checkpoint: $BEST_43"
MODEL_PATHS+=("$BEST_43")

# 3. Resume training Seed 44 từ checkpoint epoch 200
SEED=44
echo "--------------------------------------------------------"
echo "[STAGE 2/3 - RESUME] Fine-Tuning classification with Seed: $SEED"
echo "Resuming from epoch 200..."
echo "Start Time: $(date)"
echo "Logs will be appended to: logs/${ENSEMBLE_RUN_ID}/seed_${SEED}.log"
echo "--------------------------------------------------------"

$PYTHON_EXEC train.py \
    --dataset multivsl200 \
    --data-dir "$DATA_DIR" \
    --pretrain-path "checkpoints/${ENSEMBLE_RUN_ID}/seed_${SEED}/pretrained_ha_gct.pth" \
    --seed "$SEED" \
    --model-type earlyfusion \
    --d-model 256 \
    --num-classes 199 \
    --epochs 1000 \
    --patience 40 \
    --batch-size 32 \
    --accum-steps 4 \
    --lr 2e-4 \
    --classifier-lr-mult 5.0 \
    --mixup-alpha 0.4 \
    --warmup-epochs 10 \
    --drop-path-max 0.3 \
    --dropout 0.4 \
    --label-smoothing 0.15 \
    --crop-min-ratio 0.5 \
    --class-balanced \
    --loss-fn focal \
    --save-dir "checkpoints/${ENSEMBLE_RUN_ID}/seed_${SEED}/" \
    --resume "checkpoints/${ENSEMBLE_RUN_ID}/seed_${SEED}/20260622_163946/ha_gct_epoch_200.pth" \
    --use-wandb \
    >> "logs/${ENSEMBLE_RUN_ID}/seed_${SEED}.log" 2>&1

echo "Finished training for Seed $SEED at $(date)"

# Tìm checkpoint tốt nhất của Seed 44 sau khi train xong
BEST_44=$(find "checkpoints/${ENSEMBLE_RUN_ID}/seed_${SEED}/" -name "best_ha_gct_model.pth" | sort | tail -n 1)
if [ -z "$BEST_44" ] || [ ! -f "$BEST_44" ]; then
    echo "ERROR: Best checkpoint not found for Seed $SEED!"
    exit 1
fi
echo "Captured Best Checkpoint for Seed 44: $BEST_44"
MODEL_PATHS+=("$BEST_44")

# 4. Chạy Stage 3: Ensemble Evaluation với TTA
export MODEL1_PATH="${MODEL_PATHS[0]}"
export MODEL2_PATH="${MODEL_PATHS[1]}"
export MODEL3_PATH="${MODEL_PATHS[2]}"

echo "========================================================"
echo "[STAGE 3/3] Running Ensemble Evaluation with TTA"
echo "Start Time: $(date)"
echo "Model 1: $MODEL1_PATH"
echo "Model 2: $MODEL2_PATH"
echo "Model 3: $MODEL3_PATH"
echo "========================================================"

$PYTHON_EXEC -c '
import os
import sys
import torch
import torch.nn.functional as F

sys.path.extend(["./", "../"])

from utils.preprocessing import SkeletonTransforms
from data.dataloader import get_multivsl_loaders
from models.ha_gct import EarlyFusionHA_GCT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Ensemble Evaluation Device: {device}")

model_paths = [
    os.environ.get("MODEL1_PATH"),
    os.environ.get("MODEL2_PATH"),
    os.environ.get("MODEL3_PATH")
]
model_paths = [p for p in model_paths if p and os.path.exists(p)]

if not model_paths:
    raise ValueError("No valid model paths found!")

print(f"Loading {len(model_paths)} models...")

def load_model(path):
    model = EarlyFusionHA_GCT(
        num_joints=27,
        in_channels=2,
        d_model=256,
        num_ha_gc_blocks=3,
        num_mhsa_layers=2,
        nhead=4,
        num_classes=199,
        dropout=0.1,       # dropout=0 khi eval (model.eval() tự tắt)
        graph_lambda=0.05,
        max_frames=150,
        drop_path_max=0.0  # tắt drop path khi eval
    ).to(device)

    checkpoint = torch.load(path, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model

models = [load_model(p) for p in model_paths]
print(f"Loaded {len(models)} models successfully.")

# Test-Time Augmentation
def tta_predict(model, x, mask):
    B, C, T, V = x.shape

    def resample(data, rate):
        new_T = max(2, int(T * rate))
        flat = data.permute(0, 1, 3, 2).reshape(B, C * V, T)
        resampled = F.interpolate(
            F.interpolate(flat, size=new_T, mode="linear", align_corners=False),
            size=T, mode="linear", align_corners=False
        )
        return resampled.reshape(B, C, V, T).permute(0, 1, 3, 2)

    with torch.cuda.amp.autocast():
        # 1. Original
        p1 = F.softmax(model(x, mask=mask), dim=-1)

        # 2. Flip x-axis (mirror signing)
        x_flip = x.clone()
        x_flip[:, 0, :, :] = -x_flip[:, 0, :, :]
        p2 = F.softmax(model(x_flip, mask=mask), dim=-1)

        # 3. Speed x0.9
        p3 = F.softmax(model(resample(x, 0.9), mask=mask), dim=-1)

        # 4. Speed x1.1
        p4 = F.softmax(model(resample(x, 1.1), mask=mask), dim=-1)

        # 5. Speed x0.8 (thêm augmentation mạnh hơn)
        p5 = F.softmax(model(resample(x, 0.8), mask=mask), dim=-1)

    # Weighted average: original và flip quan trọng hơn
    return (2*p1 + 2*p2 + p3 + p4 + p5) / 8.0

transform = SkeletonTransforms(num_joints=27, max_frames=150, verbose=False)

_, _, test_loader = get_multivsl_loaders(
    "/mnt/nvme2/users/utbt_sv1/data/MultiVSL200/raw_npy",
    batch_size=16,   # nhỏ hơn vì TTA x5 tốn memory
    num_workers=4,
    transform=transform,
    split_method="random"
)

correct_top1 = 0
correct_top5 = 0
total = 0

print("Running ensemble + TTA evaluation...")
with torch.no_grad():
    for i, (batch_data, batch_mask, batch_labels) in enumerate(test_loader):
        batch_data  = batch_data.to(device)
        batch_mask  = batch_mask.to(device)
        batch_labels = batch_labels.to(device)

        # Trung bình xác suất của tất cả model + TTA
        probs = torch.stack([tta_predict(m, batch_data, batch_mask) for m in models]).mean(dim=0)

        num_classes = probs.size(1)
        maxk = min(5, num_classes)
        _, pred = probs.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(batch_labels.view(-1, 1).expand_as(pred))

        correct_top1 += correct[:, :1].reshape(-1).float().sum().item()
        correct_top5 += correct[:, :maxk].reshape(-1).float().sum().item()
        total += batch_labels.size(0)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_loader)}] Running Top-1: {100.*correct_top1/total:.2f}%")

top1 = 100.0 * correct_top1 / total
top5 = 100.0 * correct_top5 / total

print("\n" + "=" * 50)
print("FINAL ENSEMBLE + TTA RESULTS")
print("=" * 50)
print(f"Total test samples : {total}")
print(f"Top-1 Accuracy     : {top1:.2f}%")
print(f"Top-5 Accuracy     : {top5:.2f}%")
print("=" * 50)
' 2>&1 | tee "logs/${ENSEMBLE_RUN_ID}/ensemble_eval.log"

echo "========================================================"
echo "Pipeline finished at $(date)"
echo "========================================================"
