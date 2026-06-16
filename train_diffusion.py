import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data.dataloader import VSLDataset
from utils.skeleton_diffusion import SkeletonDiffusion

def train_diffusion_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_dataset = VSLDataset(
        'data/400VSL/processed/27_direct/train_data_joint.npy',
        'data/400VSL/processed/27_direct/train_label.pkl',
        is_train=False  # Không cần augmentation khi train diffusion
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Create Diffusion Model
    model = SkeletonDiffusion(
        num_joints=27,
        in_channels=2,
        num_frames=64,
        hidden_dim=256,
        num_timesteps=1000
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    
    print(f"Training Skeleton Diffusion Model...")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            loss = model(data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/skeleton_diffusion_epoch_{epoch+1}.pth')
    
    # Generate new data
    print("\nGenerating new skeleton sequences...")
    model.eval()
    with torch.no_grad():
        new_samples = model.sample(num_samples=100, device=device)
        
    # Save generated data
    new_samples_np = new_samples.cpu().numpy()
    np.save('data/400VSL/processed/27_direct/generated_data.npy', new_samples_np)
    print(f"Saved 100 generated sequences to generated_data.npy")

if __name__ == '__main__':
    train_diffusion_model()