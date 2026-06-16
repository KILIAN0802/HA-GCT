import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from models.ha_gct import HA_GCT

def parse_args():
    parser = argparse.ArgumentParser(description="HA-GCT Training Pipeline")
    parser.add_argument('--data-dir', type=str, default='data/400VSL/processed/27_direct', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num-classes', type=int, default=400, help='Number of action classes')
    parser.add_argument('--num-point', type=int, default=27, help='Number of skeleton joints')
    parser.add_argument('--num-person', type=int, default=2, help='Number of persons in skeleton')
    parser.add_argument('--in-channels', type=int, default=2, help='Number of coordinates (x, y)')
    parser.add_argument('--dummy-test', action='store_true', help='Force training with dummy data for verification')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='results/logs', help='Directory for TensorBoard logs')
    
    # wandb & dataset options
    parser.add_argument('--use-wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', type=str, default='HA-GCT', help='Weights & Biases project name')
    parser.add_argument('--dataset', type=str, default='vsl400', choices=['vsl400', 'multivsl200'], help='Dataset selection')
    parser.add_argument('--split-method', type=str, default='random', choices=['random', 'signer'], help='MultiVSL200 dataset split method')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume training from')
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

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for batch_data, batch_labels in progress_bar:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
        
    return total_loss / len(loader), correct / total

@torch.no_grad()
def eval_model(model, loader, criterion, device, desc="[Val]"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc=desc)
    for batch_data, batch_labels in progress_bar:
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()
        
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{100. * correct / total:.2f}%"
        })
        
    return total_loss / len(loader), correct / total

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Generate unique run ID based on timestamp or extract from checkpoint if resuming
    import datetime
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    is_resume = False
    
    if args.resume and os.path.isfile(args.resume):
        is_resume = True
        # Try to extract run_id from parent directory of the checkpoint file
        parent_dir = os.path.basename(os.path.dirname(args.resume))
        # E.g. checkpoints/20260616_221935/best_ha_gct_model.pth -> parent_dir is '20260616_221935'
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
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
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
                args.data_dir, batch_size=args.batch_size, num_workers=4, transform=transform, split_method=args.split_method
            )
        else:
            print(f"Loading 400VSL dataloaders from {args.data_dir} with SkeletonTransforms...")
            train_loader, val_loader, test_loader = get_dataloaders(
                args.data_dir, batch_size=args.batch_size, num_workers=4, transform=transform
            )
    else:
        train_loader, val_loader, test_loader = get_dummy_loaders(args)
        
    # Build model
    print("Building HA-GCT model...")
    model = HA_GCT(
        num_joints=args.num_point,
        in_channels=args.in_channels,
        d_model=256,
        num_ha_gc_blocks=3,
        num_mhsa_layers=2,
        nhead=8,
        num_classes=args.num_classes,
        dropout=0.1,
        graph_lambda=0.1,
        max_frames=max_frames
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    best_acc = -1.0
    
    if is_resume:
        print(f"Loading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Restored optimizer state.")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Restored learning rate scheduler state.")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"Resuming training from epoch {start_epoch + 1}")
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                print(f"Restored best validation accuracy: {best_acc * 100:.2f}%")
        else:
            model.load_state_dict(checkpoint)
            # Try parsing epoch from filename e.g. ha_gct_epoch_10.pth
            filename = os.path.basename(args.resume)
            if 'epoch_' in filename:
                try:
                    parts = filename.split('_')
                    epoch_part = [p for p in parts if p.isdigit() or (p.replace('.pth', '').isdigit())][0]
                    start_epoch = int(epoch_part.replace('.pth', ''))
                    print(f"Parsed epoch from filename. Resuming training from epoch {start_epoch + 1}")
                except Exception:
                    pass
            # Fast-forward scheduler if no state dict is loaded
            for _ in range(start_epoch):
                scheduler.step()
                
    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)
    writer = SummaryWriter(run_log_dir)
    
    print("\nStarting training loops...")
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device, desc=f"Epoch {epoch+1} [Val]")
        
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0]
            })
            
        print(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(run_save_dir, 'best_ha_gct_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, best_path)
            print(f"New best model saved with Val Acc: {best_acc*100:.2f}%")
            if use_wandb:
                wandb.save(best_path)
            
        # Periodic save (save full training state for resume capability)
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(run_save_dir, f'ha_gct_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
            }, periodic_path)
            print(f"Periodic checkpoint saved: {periodic_path}")
            if use_wandb:
                wandb.save(periodic_path)
            
    print("\nTraining completed. Evaluating on test set...")
    best_checkpoint = torch.load(os.path.join(run_save_dir, 'best_ha_gct_model.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc = eval_model(model, test_loader, criterion, device, desc="[Test]")
    print(f"Final Test Result - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")
    
    if use_wandb:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_acc
        })
        wandb.finish()
        
    writer.close()

if __name__ == '__main__':
    main()
