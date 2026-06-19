import argparse
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

sys.path.extend(['./', '../'])

from data.dataloader import get_dataloaders
from models.ha_gct import HA_GCT, MultiStreamHA_GCT

def parse_args():
    parser = argparse.ArgumentParser(description="HA-GCT Training Pipeline")
    parser.add_argument('--data-dir', type=str, default='data/400VSL/processed/27_direct', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping (number of epochs with no improvement)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--num-classes', type=int, default=400, help='Number of action classes')
    parser.add_argument('--num-point', type=int, default=27, help='Number of skeleton joints')
    parser.add_argument('--num-person', type=int, default=2, help='Number of persons in skeleton')
    parser.add_argument('--in-channels', type=int, default=2, help='Number of coordinates (x, y)')
    parser.add_argument('--dummy-test', action='store_true', help='Force training with dummy data for verification')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='results/logs', help='Directory for TensorBoard logs')
    
    # wandb & dataset options
    parser.add_argument('--no-wandb', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='HA-GCT', help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', type=str, default='', help='Weights & Biases entity (username or team)')
    parser.add_argument('--dataset', type=str, default='vsl400', choices=['vsl400', 'multivsl200'], help='Dataset selection')
    parser.add_argument('--split-method', type=str, default='random', choices=['random', 'signer'], help='MultiVSL200 dataset split method')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--tta', action='store_true', help='Use Test-Time Augmentation (TTA) during evaluation')
    
    # Phase 3 options
    parser.add_argument('--pretrain', action='store_true', help='Run self-supervised pre-training (Stage 1)')
    parser.add_argument('--pretrain-epochs', type=int, default=50, help='Number of pre-training epochs')
    # parser.add_argument('--pretrain-path', type=str, default='checkpoints/pretrained_ha_gct.pth', help='Path to save/load pre-trained encoder weights')
    parser.add_argument('--pretrain-path', type=str, default='', help='Path to load pre-trained encoder weights')
    parser.add_argument('--class-balanced', action='store_true', help='Use WeightedRandomSampler for class balanced sampling')
    parser.add_argument('--loss-fn', type=str, default='ce', choices=['ce', 'focal'], help='Loss function selection')
    
    # Phase 4 options
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--crop-min-ratio', type=float, default=0.6, help='Crop min ratio for spatial augmentation')
    parser.add_argument('--d-model', type=int, default=128, help='d_model dimension')
    parser.add_argument('--model-type', type=str, default='multistream', choices=['multistream', 'earlyfusion'], help='Model architecture selection')
    parser.add_argument('--mixup-alpha', type=float, default=0.1, help='Alpha parameter for Mixup augmentation (0.0 to disable)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for CrossEntropyLoss')
    parser.add_argument('--classifier-lr-mult', type=float, default=2.0, help='Classifier LR multiplier')
    parser.add_argument('--drop-path-max', type=float, default=0.0, help='Maximum DropPath probability')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--overfit-one-batch', action='store_true', help='Sanity check: train on a single batch for 500 steps to check convergence')
    
    return parser.parse_args()

def check_dataset_exists(data_dir):
    required_files = [
        'train_data_joint.npy', 'train_label.pkl',
        'val_data_joint.npy', 'val_label.pkl',
        'test_data_joint.npy', 'test_label.pkl'
    ]
    for f in required_files:
        if not os.path.exists(os.path.join(data_dir, f)):
            return False
    return True

def get_dummy_loaders(args):
    print("WARNING: Dataset files not found. Using generated dummy data for self-test...")
    
    # Generate random dummy data: (N, C, T, V, M)
    # 400VSL frames are typically around 64 frames
    num_train, num_val, num_test = 64, 16, 16
    
    train_data = np.random.randn(num_train, args.in_channels, 64, args.num_point).astype(np.float32)
    train_labels = np.random.randint(0, args.num_classes, num_train)
    
    val_data = np.random.randn(num_val, args.in_channels, 64, args.num_point).astype(np.float32)
    val_labels = np.random.randint(0, args.num_classes, num_val)
    
    test_data = np.random.randn(num_test, args.in_channels, 64, args.num_point).astype(np.float32)
    test_labels = np.random.randint(0, args.num_classes, num_test)
    
    # Create simple tensor datasets
    train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
    val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.LongTensor(val_labels))
    test_dataset = TensorDataset(torch.FloatTensor(test_data), torch.LongTensor(test_labels))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
    
    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def generate_mask_exact(batch_size, num_frames, num_joints, device):
    # Vectorized exact masking of 50-75% of joints per frame
    rand = torch.rand(batch_size, num_frames, num_joints, device=device)
    ids_shuffle = torch.argsort(rand, dim=-1)
    
    p = 0.5 + 0.25 * torch.rand(batch_size, num_frames, 1, device=device)
    num_mask = (p * num_joints).long()
    
    ranks = torch.arange(num_joints, device=device).view(1, 1, -1).expand(batch_size, num_frames, -1)
    mask = ranks < num_mask
    
    real_mask = torch.zeros_like(mask).scatter_(-1, ids_shuffle, mask)
    return real_mask

class MaskedSkeletonAutoencoder(nn.Module):
    def __init__(self, encoder, d_model, max_frames, in_channels=2):
        super().__init__()
        self.encoder = encoder
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, max_frames * in_channels)
        )
        self.max_frames = max_frames
        self.in_channels = in_channels
        
    def forward(self, x, mask):
        B, C, T, N = x.shape
        
        # STEP 1: Physical Embedding (takes (B, C, T, N) -> returns (B, T, N, D))
        x_embed = self.encoder.physical_embedding(x)  # (B, T, N, D)
        
        # Mask the joint embeddings
        mask_expanded = mask.unsqueeze(-1)  # (B, T, N, 1)
        x_embed = torch.where(mask_expanded, self.mask_token, x_embed)
        
        # STEP 2: Spatial Branch (HA-GC x3)
        # HA-GC expects shape (B, D, T, N)
        x_spatial = x_embed.permute(0, 3, 1, 2).contiguous()  # (B, D, T, N)
        for ha_gc_block in self.encoder.spatial_branch:
            x_spatial = ha_gc_block(x_spatial)
        x_spatial = x_spatial + self.encoder.spatial_temporal_conv(x_spatial)
        x_spatial = 0.5 * (x_spatial.mean(dim=2) + x_spatial.max(dim=2)[0])
        x_spatial = x_spatial.transpose(1, 2)  # (B, N, D)
        
        # STEP 3: Temporal Branch (MHSA x2)
        # Learnable joint attention pooling
        joint_score = self.encoder.joint_pool_score(x_embed)          # (B, T, N, 1)
        joint_weight = torch.softmax(joint_score, dim=2)              # (B, T, N, 1)
        x_temporal = (x_embed * joint_weight).sum(dim=2)              # (B, T, D)
        
        # Local temporal modeling (Depthwise Conv1D)
        x_conv = x_temporal.transpose(1, 2)
        x_conv = self.encoder.local_temporal_conv(x_conv)
        x_temporal = x_temporal + x_conv.transpose(1, 2)
        x_temporal = self.encoder.local_temporal_norm(x_temporal)
        
        x_temporal = self.encoder.temporal_proj(x_temporal)
        
        # MHSA Layer (standard dot-product attention)
        for mhsa_layer in self.encoder.temporal_branch:
            x_temporal, attn_weights = mhsa_layer(x_temporal)
            
        # STEP 4: Gated Fusion & LayerNorm
        x_temporal_mean = x_temporal.mean(dim=1, keepdim=True)  # (B, 1, D)
        x_temporal_expanded = x_temporal_mean.expand(-1, x_spatial.size(1), -1)  # (B, N, D)
        
        gate_input = self.encoder.fusion_norm(torch.cat([
            x_spatial,
            x_temporal_expanded
        ], dim=-1))
        gate = torch.sigmoid(self.encoder.fusion_gate(gate_input))
        
        x_fused = gate * x_spatial + (1 - gate) * x_temporal_expanded
        x_fused = self.encoder.post_fusion_norm(x_fused)  # (B, N, D)
        
        # STEP 5: Decode to coordinates
        out = self.decoder(x_fused)  # (B, N, T_max * C)
        out = out.view(B, N, self.max_frames, self.in_channels)
        out = out[:, :, :T, :]
        out = out.permute(0, 2, 1, 3).contiguous()  # (B, T, N, C)
        
        return out

def pretrain_epoch(autoencoder, loader, optimizer, scheduler, device, epoch, scaler=None):
    autoencoder.train()
    total_loss = 0.0
    use_amp = (device.type == 'cuda' and scaler is not None)
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Pre-train]")
    for step, (batch_data, _) in enumerate(progress_bar):
        batch_data = batch_data.to(device)
        B, C, T, N = batch_data.shape
        
        mask = generate_mask_exact(B, T, N, device)
        x_gt = batch_data.permute(0, 2, 3, 1).contiguous()
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                x_pred = autoencoder(batch_data, mask)
                mask_expanded = mask.unsqueeze(-1).expand_as(x_gt)
                loss = F.mse_loss(x_pred[mask_expanded], x_gt[mask_expanded])
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        else:
            x_pred = autoencoder(batch_data, mask)
            mask_expanded = mask.unsqueeze(-1).expand_as(x_gt)
            loss = F.mse_loss(x_pred[mask_expanded], x_gt[mask_expanded])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.6f}"})
        
    return total_loss / len(loader)

def load_pretrained_encoder(model, pretrained_path, device):
    print(f"Loading pre-trained encoder weights from {pretrained_path}...")
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        pretrained_dict = checkpoint['model_state_dict']
    else:
        pretrained_dict = checkpoint
        
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('classifier')}
    model_dict = model.state_dict()
    
    for stream_name in ['stream_joint', 'stream_bone', 'stream_velocity']:
        mapped_dict = {}
        skipped = []
        for k, v in pretrained_dict.items():
            mapped_key = f"{stream_name}.{k}"
            if mapped_key in model_dict and model_dict[mapped_key].shape == v.shape:
                mapped_dict[mapped_key] = v
            else:
                skipped.append((
                    mapped_key,
                    tuple(v.shape),
                    tuple(model_dict[mapped_key].shape) if mapped_key in model_dict else None
                ))
                
        if mapped_dict:
            model.load_state_dict(mapped_dict, strict=False)
            print(f"  Initialized {stream_name} with {len(mapped_dict)} matched tensors.")
        else:
            print(f"  Skipped {stream_name}: no compatible pretrained tensors.")
            
        if skipped and len(mapped_dict) > 0:
            print(f"  Skipped {len(skipped)} incompatible/missing tensors for {stream_name}.")

def get_dataset_labels(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        base_dataset = dataset.dataset
        base_labels = get_dataset_labels(base_dataset)
        return [base_labels[i] for i in dataset.indices]
        
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return dataset.tensors[1].tolist()
        
    if hasattr(dataset, 'labels'):
        return list(dataset.labels)
        
    if hasattr(dataset, 'files'):
        return [item[1] for item in dataset.files]
        
    raise ValueError("Could not extract labels from dataset of type " + str(type(dataset)))

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def topk_accuracy_count(output, target, topk=(1, 5)):
    with torch.no_grad():
        num_classes = output.size(1)
        actual_topk = [k for k in topk if k <= num_classes]
        if not actual_topk:
            return [0.0] * len(topk)
        
        maxk = max(actual_topk)
        pred = output.topk(maxk, dim=1, largest=True, sorted=True).indices
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        res = []
        for k in topk:
            if k > num_classes:
                res.append(float(target.size(0)))
            else:
                correct_k = correct[:, :k].reshape(-1).float().sum(0).item()
                res.append(correct_k)
        return res

def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, scaler=None, mixup_alpha=0.0, accum_steps=4):
    model.train()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    use_amp = (device.type == 'cuda' and scaler is not None)
    
    # Thiết lập tham số Mixup
    ACCUM_STEPS = accum_steps  # Tích lũy gradient
    
    optimizer.zero_grad()
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for step, (batch_data, batch_labels) in enumerate(progress_bar):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        # ==========================================
        # MIXUP AUGMENTATION
        # ==========================================
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
        else:
            lam = 1.0
            
        # Trộn ngẫu nhiên các sample trong cùng 1 batch
        index = torch.randperm(batch_data.size(0)).to(device)
        mixed_data = lam * batch_data + (1 - lam) * batch_data[index]
        # ==========================================
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(mixed_data)
                # Tính tổng loss của 2 nhãn được trộn
                loss = lam * criterion(outputs, batch_labels) + (1 - lam) * criterion(outputs, batch_labels[index])
            
            loss = loss / ACCUM_STEPS
            scaler.scale(loss).backward()
            
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
        else:
            outputs = model(mixed_data)
            loss = lam * criterion(outputs, batch_labels) + (1 - lam) * criterion(outputs, batch_labels[index])
            
            loss = loss / ACCUM_STEPS
            loss.backward()
            
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        loss_val = loss.item() * ACCUM_STEPS
        total_loss += loss_val
        
        # Tính Accuracy (tính trên nhãn chiếm tỷ trọng lớn hơn lam > 0.5)
        target_label = batch_labels if lam >= 0.5 else batch_labels[index]
        
        top1_c, top5_c = topk_accuracy_count(outputs, target_label)
        correct_top1 += top1_c
        correct_top5 += top5_c
        total += batch_labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss_val:.4f}",
            'acc_top1': f"{100. * correct_top1 / total:.2f}%",
            'acc_top5': f"{100. * correct_top5 / total:.2f}%"
        })
        
    return total_loss / len(loader), correct_top1 / total, correct_top5 / total

@torch.no_grad()
def eval_model(model, loader, criterion, device, desc="[Val]", use_tta=False):
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    use_amp = (device.type == 'cuda')
    
    def perturb_speed(batch, rate):
        B, C, T, V = batch.shape
        x = batch.permute(0, 1, 3, 2).reshape(B, C * V, T)
        new_T = max(2, int(T * rate))
        x_resampled = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
        x_final = F.interpolate(x_resampled, size=T, mode='linear', align_corners=False)
        return x_final.reshape(B, C, V, T).permute(0, 1, 3, 2)
    
    progress_bar = tqdm(loader, desc=desc)
    for batch_data, batch_labels in progress_bar:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                if use_tta:
                    out_orig = model(batch_data)
                    
                    batch_flipped = batch_data.clone()
                    batch_flipped[:, 0, :, :] = -batch_flipped[:, 0, :, :]
                    out_flipped = model(batch_flipped)
                    
                    batch_speed_09 = perturb_speed(batch_data, 0.9)
                    out_speed_09 = model(batch_speed_09)
                    
                    batch_speed_11 = perturb_speed(batch_data, 1.1)
                    out_speed_11 = model(batch_speed_11)
                    
                    prob_orig = F.softmax(out_orig, dim=-1)
                    prob_flipped = F.softmax(out_flipped, dim=-1)
                    prob_speed_09 = F.softmax(out_speed_09, dim=-1)
                    prob_speed_11 = F.softmax(out_speed_11, dim=-1)
                    
                    prob_avg = (prob_orig + prob_flipped + prob_speed_09 + prob_speed_11) / 4.0
                    outputs = torch.log(torch.clamp(prob_avg, min=1e-12))
                else:
                    outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
        else:
            if use_tta:
                out_orig = model(batch_data)
                
                batch_flipped = batch_data.clone()
                batch_flipped[:, 0, :, :] = -batch_flipped[:, 0, :, :]
                out_flipped = model(batch_flipped)
                
                batch_speed_09 = perturb_speed(batch_data, 0.9)
                out_speed_09 = model(batch_speed_09)
                
                batch_speed_11 = perturb_speed(batch_data, 1.1)
                out_speed_11 = model(batch_speed_11)
                
                prob_orig = F.softmax(out_orig, dim=-1)
                prob_flipped = F.softmax(out_flipped, dim=-1)
                prob_speed_09 = F.softmax(out_speed_09, dim=-1)
                prob_speed_11 = F.softmax(out_speed_11, dim=-1)
                
                prob_avg = (prob_orig + prob_flipped + prob_speed_09 + prob_speed_11) / 4.0
                outputs = torch.log(torch.clamp(prob_avg, min=1e-12))
            else:
                outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
        
        total_loss += loss.item()
        top1_c, top5_c = topk_accuracy_count(outputs, batch_labels)
        correct_top1 += top1_c
        correct_top5 += top5_c
        total += batch_labels.size(0)
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc_top1': f"{100. * correct_top1 / total:.2f}%",
            'acc_top5': f"{100. * correct_top5 / total:.2f}%"
        })
        
    return total_loss / len(loader), correct_top1 / total, correct_top5 / total

def main():
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"GPU Memory Status: Free: {free_mem / (1024**3):.2f} GB / Total: {total_mem / (1024**3):.2f} GB")
        except Exception as e:
            print(f"Could not retrieve GPU memory info: {e}")
    
    # Generate unique run ID based on timestamp or extract from checkpoint if resuming
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    is_resume = False
    
    if args.resume and os.path.isfile(args.resume):
        is_resume = True
        parent_dir = os.path.basename(os.path.dirname(args.resume))
        if parent_dir and parent_dir != 'checkpoints':
            run_id = parent_dir
            print(f"Resuming run with ID: {run_id}")
            
    run_save_dir = os.path.join(args.save_dir, run_id)
    run_log_dir = os.path.join(args.log_dir, run_id)
    
    # Determine max frames based on dataset
    max_frames = 150 if args.dataset == 'multivsl200' else 64
    
    # Auto-adjust classes for MultiVSL200 if not customized
    if args.dataset == 'multivsl200' and args.num_classes == 400:
        args.num_classes = 199
        print(f"Auto-configured num_classes to 199 for MultiVSL200 dataset.")
        
    # Initialize wandb if requested and available
    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    
    if not args.no_wandb and not WANDB_AVAILABLE:
        print("=" * 70)
        print("WARNING: Weights & Biases (wandb) is not installed or import failed.")
        print("         Logs will NOT be sent to wandb.")
        print("         To resolve this, please run:")
        print("             pip install wandb")
        print("         And ensure you are logged in using 'wandb login'.")
        print("=" * 70)
        
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity if args.wandb_entity else None,
            config=vars(args),
            name=f"ha_gct_{args.dataset}_{run_id}",
            id=run_id,
            resume="allow"
        )
        print(f"Initialized Weights & Biases (wandb) logger for run: {run_id}")
    
    # Load loaders
    if not args.dummy_test:
        from utils.preprocessing import SkeletonTransforms
        
        transform = SkeletonTransforms(
            num_joints=args.num_point,
            max_frames=max_frames,
            verbose=False
        )
        
        if args.dataset == 'multivsl200':
            print(f"Loading MultiVSL200 dataloaders from {args.data_dir} with SkeletonTransforms...")
            from data.dataloader import get_multivsl_loaders
            train_loader, val_loader, test_loader = get_multivsl_loaders(
                args.data_dir, batch_size=args.batch_size, num_workers=4, transform=transform, split_method=args.split_method, crop_min_ratio=args.crop_min_ratio
            )
        else:
            print(f"Loading 400VSL dataloaders from {args.data_dir} with SkeletonTransforms...")
            train_loader, val_loader, test_loader = get_dataloaders(
                args.data_dir, batch_size=args.batch_size, num_workers=4, transform=transform, crop_min_ratio=args.crop_min_ratio
            )
    else:
        train_loader, val_loader, test_loader = get_dummy_loaders(args)
        
    # GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # =========================================================================
    # STAGE 1: Self-Supervised Pre-Training (Masked Skeleton Autoencoder - MSA)
    # =========================================================================
    if args.pretrain:
        print("=" * 70)
        print("STAGE 1: SELF-SUPERVISED PRE-TRAINING (MASKED SKELETON AUTOENCODER)")
        print("=" * 70)
        
        # Combine train & val datasets for pre-training (as val has no supervision anyway)
        from torch.utils.data import ConcatDataset
        combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
        
        dataloader_kwargs = {
            'batch_size': args.batch_size,
            'num_workers': train_loader.num_workers,
            'pin_memory': train_loader.pin_memory
        }
        if train_loader.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = train_loader.prefetch_factor
            
        combined_loader = DataLoader(
            combined_dataset,
            shuffle=True,
            **dataloader_kwargs
        )
        
        # Build base single-stream model
        print(f"Building single-stream HA-GCT encoder with d_model={args.d_model}...")
        encoder = HA_GCT(
            num_joints=args.num_point,
            in_channels=args.in_channels,
            d_model=args.d_model,
            num_ha_gc_blocks=3,
            num_mhsa_layers=2,
            nhead=4,
            dropout=0.1,  # Lower dropout for pre-training
            graph_lambda=0.1,
            max_frames=max_frames,
            drop_path_max=args.drop_path_max
        ).to(device)
        
        autoencoder = MaskedSkeletonAutoencoder(
            encoder=encoder,
            d_model=args.d_model,
            max_frames=max_frames,
            in_channels=args.in_channels
        ).to(device)
        
        optimizer = optim.AdamW(autoencoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        steps_per_epoch = len(combined_loader)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.pretrain_epochs * steps_per_epoch, eta_min=1e-6
        )
        
        os.makedirs(os.path.dirname(args.pretrain_path), exist_ok=True)
        best_loss = float('inf')
        
        print("\nStarting self-supervised pre-training loops...")
        for epoch in range(args.pretrain_epochs):
            train_loss = pretrain_epoch(autoencoder, combined_loader, optimizer, scheduler, device, epoch, scaler)
            print(f"Epoch {epoch+1}/{args.pretrain_epochs} - MSA Loss: {train_loss:.6f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    "pretrain_epoch": epoch + 1,
                    "pretrain_loss": train_loss,
                    "pretrain_lr": scheduler.get_last_lr()[0]
                })
                
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': autoencoder.encoder.state_dict(),
                    'loss': best_loss
                }, args.pretrain_path)
                print(f"Saved best pre-trained encoder weights to {args.pretrain_path} with loss {best_loss:.6f}")
                
        print("\nSelf-supervised pre-training completed successfully!")
        if use_wandb:
            wandb.finish()
        return

    # =========================================================================
    # STAGE 2: Fine-Tuning classification
    # =========================================================================
    # Class-Balanced Sampling
    if args.class_balanced:
        print("Enabling Class-Balanced Sampling using WeightedRandomSampler...")
        labels = get_dataset_labels(train_loader.dataset)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        unique_labels, counts = torch.unique(labels_tensor, return_counts=True)
        max_class_id = int(torch.max(labels_tensor).item())
        
        class_weights = torch.zeros(max_class_id + 1, dtype=torch.float)
        for l, c in zip(unique_labels, counts):
            class_weights[l] = 1.0 / float(c)
            
        sample_weights = class_weights[labels_tensor]
        
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        dataloader_kwargs = {
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers,
            'pin_memory': train_loader.pin_memory,
        }
        if train_loader.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = train_loader.prefetch_factor
            
        train_loader = DataLoader(
            train_loader.dataset,
            sampler=sampler,
            **dataloader_kwargs
        )

    # Build classification model
    print(f"Building model type: {args.model_type} with d_model={args.d_model}...")
    from models.ha_gct import EarlyFusionHA_GCT
    if args.model_type == 'earlyfusion':
        model = EarlyFusionHA_GCT(
            num_joints=args.num_point,
            in_channels=args.in_channels,
            d_model=args.d_model,
            num_ha_gc_blocks=3,
            num_mhsa_layers=2,
            nhead=4,
            num_classes=args.num_classes,
            dropout=args.dropout,
            graph_lambda=0.05,
            max_frames=max_frames,
            drop_path_max=args.drop_path_max
        ).to(device)
    else:
        model = MultiStreamHA_GCT(
            num_joints=args.num_point,
            in_channels=args.in_channels,
            d_model=args.d_model,
            num_ha_gc_blocks=3,
            num_mhsa_layers=2,
            nhead=4,
            num_classes=args.num_classes,
            dropout=args.dropout,
            graph_lambda=0.05,
            max_frames=max_frames,
            drop_path_max=args.drop_path_max
        ).to(device)
    
    # Load pre-trained weights if provided and exists
    pretrained_loaded = False
    if args.pretrain_path and os.path.exists(args.pretrain_path):
        load_pretrained_encoder(model, args.pretrain_path, device)
        pretrained_loaded = True
    else:
        print("No pre-trained encoder found or specified. Training from scratch.")
        
    # Set Loss Function
    if args.loss_fn == 'focal':
        print("Using Focal Loss instead of CrossEntropyLoss...")
        criterion = FocalLoss(gamma=2.0, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        
    # Set up optimizer with separate learning rates if pre-trained weights were loaded
    if pretrained_loaded:
        encoder_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                encoder_params.append(param)
        
        encoder_lr = args.lr
        classifier_lr = args.lr * args.classifier_lr_mult

        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': classifier_params, 'lr': classifier_lr}
        ], weight_decay=args.weight_decay)

        print(f"Configured separate optimizer learning rates: Encoder lr = {encoder_lr:g}, Classifier lr = {classifier_lr:g}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Sequential learning rate scheduler with Linear Warm-up & Cosine Annealing Warm Restarts (Step-based)
    ACCUM_STEPS = args.accum_steps
    steps_per_epoch = max(1, (len(train_loader) + ACCUM_STEPS - 1) // ACCUM_STEPS)
    
    warmup_epochs = args.warmup_epochs
    if args.dummy_test:
        warmup_epochs = min(2, args.warmup_epochs)
        
    warmup_steps = warmup_epochs * steps_per_epoch
    
    # Cosine Annealing Warm Restarts parameters
    T_0_epochs = 50
    T_0_steps = T_0_epochs * steps_per_epoch
    
    # start_factor to scale from 1e-6 to args.lr (default 3e-4)
    start_factor = 1e-6 / args.lr
        
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0_steps, T_mult=1, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    start_epoch = 0
    best_acc = -1.0
    patience_counter = 0
    
    if is_resume:
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Restored optimizer state.")
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Restored learning rate scheduler state.")
                except Exception as e:
                    print(f"Warning: Could not restore scheduler state dict: {e}. Re-initializing scheduler.")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                print(f"Restored best validation accuracy: {best_acc * 100:.2f}%")
            if 'patience_counter' in checkpoint:
                patience_counter = checkpoint['patience_counter']
                print(f"Restored patience counter: {patience_counter}")
        else:
            model.load_state_dict(checkpoint)
            filename = os.path.basename(args.resume)
            if 'epoch_' in filename:
                try:
                    parts = filename.split('_')
                    epoch_part = [p for p in parts if p.isdigit() or (p.replace('.pth', '').isdigit())][0]
                    start_epoch = int(epoch_part.replace('.pth', ''))
                    print(f"Parsed epoch from filename. Resuming training from epoch {start_epoch + 1}")
                except Exception:
                    pass
            for _ in range(start_epoch * steps_per_epoch):
                scheduler.step()
                
    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)
    
    # =========================================================================
    # SANITY CHECK: Overfit One Batch
    # =========================================================================
    if args.overfit_one_batch:
        print("\n" + "=" * 70)
        print("SANITY CHECK: OVERFIT ONE BATCH TEST")
        print("=" * 70)
        
        # Get one single batch from train_loader
        batch_data, batch_labels = next(iter(train_loader))
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        print(f"Batch shape: {batch_data.shape}")
        print(f"Labels: {batch_labels.cpu().tolist()}")
        
        # Re-initialize a simple optimizer with a healthy learning rate for overfitting
        overfit_lr = 1e-3
        optimizer = optim.AdamW(model.parameters(), lr=overfit_lr)
        print(f"Training on this single batch for 500 steps with AdamW (lr={overfit_lr})...")
        
        model.train()
        for step in range(500):
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            if (step + 1) % 25 == 0 or step == 0:
                top1_c, _ = topk_accuracy_count(outputs, batch_labels)
                acc = 100.0 * top1_c / batch_data.size(0)
                print(f"Step {step+1:03d}/500 - Loss: {loss.item():.6f} - Acc: {acc:.2f}%")
                
        print("Overfit test completed!")
        print("=" * 70 + "\n")
        return
        
    print("\nStarting training loops...")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc_top1, train_acc_top5 = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, scaler, mixup_alpha=args.mixup_alpha, accum_steps=args.accum_steps)
        val_loss, val_acc_top1, val_acc_top5 = eval_model(model, val_loader, criterion, device, desc=f"Epoch {epoch+1} [Val]", use_tta=args.tta)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train_top1', train_acc_top1, epoch)
        writer.add_scalar('Accuracy/train_top5', train_acc_top5, epoch)
        writer.add_scalar('Accuracy/val_top1', val_acc_top1, epoch)
        writer.add_scalar('Accuracy/val_top5', val_acc_top5, epoch)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc_top1,
                "train_acc_top5": train_acc_top5,
                "val_loss": val_loss,
                "val_acc": val_acc_top1,
                "val_acc_top5": val_acc_top5,
                "lr": scheduler.get_last_lr()[0]
            })
            
        print(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Train Acc (Top-1/Top-5): {train_acc_top1*100:.2f}%/{train_acc_top5*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc (Top-1/Top-5): {val_acc_top1*100:.2f}%/{val_acc_top5*100:.2f}%")
        
        # Save best model based on Top-1
        if val_acc_top1 > best_acc:
            best_acc = val_acc_top1
            patience_counter = 0
            best_path = os.path.join(run_save_dir, 'best_ha_gct_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'patience_counter': patience_counter,
            }, best_path)
            print(f"New best model saved with Val Acc (Top-1): {best_acc*100:.2f}%")
            if use_wandb:
                wandb.save(best_path)
        # else:
        #     patience_counter += 1
        #     print(f"Early Stopping Counter: {patience_counter}/{args.patience}")
        #     if patience_counter >= args.patience:
        #         print(f"Early stopping triggered! Training stopped after {epoch+1} epochs because Val Acc (Top-1) did not improve for {args.patience} epochs.")
        #         break
            
        # Periodic save (save full training state for resume capability)
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(run_save_dir, f'ha_gct_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'patience_counter': patience_counter,
            }, periodic_path)
            print(f"Periodic checkpoint saved: {periodic_path}")
            if use_wandb:
                wandb.save(periodic_path)
            
    print("\nTraining completed. Evaluating on test set...")
    best_checkpoint = torch.load(os.path.join(run_save_dir, 'best_ha_gct_model.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc_top1, test_acc_top5 = eval_model(model, test_loader, criterion, device, desc="[Test]", use_tta=args.tta)
    print(f"Final Test Result - Loss: {test_loss:.4f}, Acc (Top-1/Top-5): {test_acc_top1*100:.2f}%/{test_acc_top5*100:.2f}%")
    
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc_top1,
            "test_acc_top5": test_acc_top5
        })
        wandb.finish()
        
    writer.close()

if __name__ == '__main__':
    main()
