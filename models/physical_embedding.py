import torch
import torch.nn as nn
import math

class PhysicalEmbedding(nn.Module):
    """
    Physical Embedding Block for Skeleton Data (Linear Embed + Positional Embedding)
    
    Transforms raw skeleton coordinates of shape (B, T, N, C) to (B, T, N, D)
    where:
    - B: Batch size
    - T: Number of frames
    - N: Number of joints
    - C: Coordinate channels (x, y, [confidence])
    - D: Embedding dimension (d_model)
    """
    def __init__(self, in_channels, num_joints, d_model, max_frames=100, dropout=0.1):
        super().__init__()
        
        # Conv2d projection for coordinate features
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=1)
        
        # Learnable spatial positional embeddings for each joint
        self.joint_embed = nn.Parameter(torch.randn(1, 1, num_joints, d_model) * 0.02)
        
        # Learnable temporal positional embeddings for frames
        self.frame_embed = nn.Parameter(torch.randn(1, max_frames, 1, d_model) * 0.02)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor shape (B, T, N, C)
        Returns:
            out: Embedded tensor shape (B, T, N, D)
        """
        B, T, N, C = x.shape
        
        # Permute to (B, C, T, N) for Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Project coordinate channels: (B, C, T, N) -> (B, D, T, N)
        x = self.projection(x)
        
        # Permute back to (B, T, N, D)
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Add joint and frame embeddings
        # joint_embed is broadcasted across Batch and Time dimensions
        # frame_embed is broadcasted across Batch and Joint dimensions
        x = x + self.joint_embed[:, :, :N, :] + self.frame_embed[:, :T, :, :]
        
        return self.dropout(x)

if __name__ == '__main__':
    print("=" * 70)
    print("TEST PHYSICAL EMBEDDING MODULE")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    num_frames = 64
    num_joints = 27
    in_channels = 2
    d_model = 256
    
    # Dummy input: (B, T, N, C)
    x = torch.randn(batch_size, num_frames, num_joints, in_channels)
    print(f"Input shape: {x.shape} (B, T, N, C)")
    
    # Initialize module
    embed = PhysicalEmbedding(
        in_channels=in_channels,
        num_joints=num_joints,
        d_model=d_model,
        max_frames=100,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in embed.parameters()):,}")
    
    # Forward pass
    output = embed(x)
    print(f"Output shape: {output.shape} (B, T, N, D)")
    
    assert output.shape == (batch_size, num_frames, num_joints, d_model), "Incorrect output shape!"
    print("\nPHYSICAL EMBEDDING TEST PASSED!")
    print("=" * 70)
