import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhysicalEmbedding(nn.Module):
    """
    Physical Embedding Block for Skeleton Data (Linear Embed + Positional Embedding)
    
    Transforms raw skeleton coordinates of shape (B, C, T, V) to (B, T, V, D)
    where:
    - B: Batch size
    - C: Coordinate channels (x, y, [confidence])
    - T: Number of frames
    - V: Number of joints
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
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor shape (B, C, T, V)
            mask: Optional mask shape (B, T)
        Returns:
            out: Embedded tensor shape (B, T, V, D)
        """
        B, C, T, V = x.shape
        
        # Project coordinate channels: (B, C, T, V) -> (B, D, T, V)
        x_proj = self.projection(x)
        
        # Permute to (B, T, V, D)
        x_proj = x_proj.permute(0, 2, 3, 1).contiguous()
        
        # Slice frame positional embedding to sequence length T
        if T > self.frame_embed.shape[1]:
            repeats = (T + self.frame_embed.shape[1] - 1) // self.frame_embed.shape[1]
            pe_t = self.frame_embed.repeat(1, repeats, 1, 1)[:, :T, :, :]
        else:
            pe_t = self.frame_embed[:, :T, :, :]
        
        # Add joint and frame embeddings
        # joint_embed is broadcasted across Batch and Time dimensions: (1, 1, V, D)
        # pe_t is broadcasted across Batch and Joint dimensions: (1, T, 1, D)
        out = x_proj + self.joint_embed[:, :, :V, :] + pe_t
        out = self.norm(out)
        
        out = self.dropout(out)
        
        if mask is not None:
            # mask shape: (B, T) -> (B, T, 1, 1)
            out = out * mask.view(B, T, 1, 1)
            
        return out

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
    
    # Dummy input: (B, C, T, V)
    x = torch.randn(batch_size, in_channels, num_frames, num_joints)
    print(f"Input shape: {x.shape} (B, C, T, V)")
    
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
    print(f"Output shape: {output.shape} (B, T, V, D)")
    
    assert output.shape == (batch_size, num_frames, num_joints, d_model), "Incorrect output shape!"
    print("\nPHYSICAL EMBEDDING TEST PASSED!")
    print("=" * 70)
