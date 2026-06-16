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
    
    # Load loaders
    if check_dataset_exists(args.data_dir) and not args.dummy_test:
        print(f"Loading dataloaders from {args.data_dir} with SkeletonTransforms...")
        from utils.preprocessing import SkeletonTransforms
        transform = SkeletonTransforms(
            num_joints=args.num_point,
            max_frames=64,
            verbose=False
        )
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
        graph_lambda=0.1
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    
    best_acc = -1.0
    print("\nStarting training loops...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device, desc=f"Epoch {epoch+1} [Val]")
        
        scheduler.step()
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f"Epoch {epoch+1} Summary - Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, 'best_ha_gct_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, best_path)
            print(f"New best model saved with Val Acc: {best_acc*100:.2f}%")
            
        # Periodic save
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(args.save_dir, f'ha_gct_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), periodic_path)
            
    print("\nTraining completed. Evaluating on test set...")
    best_checkpoint = torch.load(os.path.join(args.save_dir, 'best_ha_gct_model.pth'), map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_loss, test_acc = eval_model(model, test_loader, criterion, device, desc="[Test]")
    print(f"Final Test Result - Loss: {test_loss:.4f}, Acc: {test_acc*100:.2f}%")
    
    writer.close()

if __name__ == '__main__':
    main()
