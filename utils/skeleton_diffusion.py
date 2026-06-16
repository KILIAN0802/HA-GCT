import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SkeletonDiffusion(nn.Module):
    """
    Skeleton Diffusion Model (DDPM)
    
    Tham khảo:
    1. Denoising Diffusion Probabilistic Models (Ho et al., NeurIPS 2020)
    2. MotionDiffuse (Zhang et al., IEEE TPAMI 2023)
    
    Cơ chế:
    - Forward Process: Thêm nhiễu Gaussian vào skeleton sequence theo từng bước.
    - Reverse Process: Học một mạng neural (Noise Predictor) để khử nhiễu, 
      từ đó sinh ra skeleton sequence mới hợp lệ về giải phẫu.
    """
    
    def __init__(
        self,
        num_joints=27,
        in_channels=2,
        num_frames=64,
        hidden_dim=128,
        num_layers=4,
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.num_timesteps = num_timesteps
        
        # ========== DIFFUSION SCHEDULE (Linear Beta Schedule) ==========
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.register_buffer('betas', betas)
        
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        # ========== NOISE PREDICTOR (U-Net 1D hoặc MLP) ==========
        # Input: (B, T, V*C) + time embedding
        # Output: Predicted noise (B, T, V*C)
        input_dim = num_frames * num_joints * in_channels
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.model = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim * 4),
            nn.GELU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim * 4, input_dim)
        )
    
    def forward(self, x, t=None):
        """
        Forward pass (Training)
        
        Args:
            x: Ground truth skeleton sequence (B, C, T, V)
            t: Timestep (B,)
        
        Returns:
            loss: MSE loss between predicted noise and actual noise
        """
        B, C, T, V = x.shape
        
        # Flatten spatial dimensions: (B, C, T, V) -> (B, T*V*C)
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, -1)
        
        # Sample random timesteps
        if t is None:
            t = torch.randint(0, self.num_timesteps, (B,), device=x.device).long()
        
        # Forward diffusion: Add noise
        noise = torch.randn_like(x_flat)
        sqrt_alpha_cumprod = torch.sqrt(self.alphas_cumprod[t]).view(B, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(B, 1)
        
        x_noisy = sqrt_alpha_cumprod * x_flat + sqrt_one_minus_alpha_cumprod * noise
        
        # Predict noise
        time_emb = self.time_mlp(t)
        model_input = torch.cat([x_noisy, time_emb], dim=-1)
        predicted_noise = self.model(model_input)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(self, num_samples=1, device='cpu'):
        """
        Generate new skeleton sequences (Sampling)
        
        Args:
            num_samples: Number of sequences to generate
        
        Returns:
            samples: Generated skeleton sequences (num_samples, C, T, V)
        """
        B = num_samples
        input_dim = self.num_frames * self.num_joints * self.in_channels
        
        # Start from pure noise
        x = torch.randn(B, input_dim, device=device)
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            time_emb = self.time_mlp(t_batch)
            model_input = torch.cat([x, time_emb], dim=-1)
            predicted_noise = self.model(model_input)
            
            # Remove noise
            alpha = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=device)
            
            beta_t = self.betas[t]
            
            # Formula: x_{t-1} = (x_t - beta_t / sqrt(1-alpha_t) * noise) / sqrt(alpha_t) + sigma * z
            x = (1.0 / torch.sqrt(alpha)) * (x - (beta_t / torch.sqrt(1 - alpha)) * predicted_noise)
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(beta_t)
                x = x + sigma * noise
        
        # Reshape back to (B, C, T, V)
        samples = x.view(B, self.num_frames, self.in_channels, self.num_joints)
        samples = samples.permute(0, 2, 1, 3)  # (B, C, T, V)
        
        return samples


class SinusoidalPositionEmbeddings(nn.Module):
    """Time embedding for Diffusion Model"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ========== TEST ==========
if __name__ == '__main__':
    print("=" * 70)
    print("TEST SKELETON DIFFUSION MODEL")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dummy data: (B=4, C=2, T=64, V=27)
    x = torch.randn(4, 2, 64, 27).to(device)
    print(f"Input shape: {x.shape}")
    
    # Create model
    model = SkeletonDiffusion(
        num_joints=27,
        in_channels=2,
        num_frames=64,
        hidden_dim=128,
        num_timesteps=100  # Small for testing
    ).to(device)
    
    print(f"Diffusion Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test training step
    loss = model(x)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test sampling (generation)
    print("\nGenerating new skeleton sequences...")
    samples = model.sample(num_samples=2, device=device)
    print(f"Generated samples shape: {samples.shape}")
    
    print("\n✅ SKELETON DIFFUSION TEST PASSED!")