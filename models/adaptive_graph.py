import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveGraphRefinement(nn.Module):
    """
    Adaptive Graph Refinement Block
    
    Generates a dynamic time-frame relationship adjacency matrix of shape (B, T, T)
    from temporal features of shape (B, T, D).
    
    Combines:
    1. A learned static/dynamic base matrix (interpolated dynamically for arbitrary T).
    2. A data-dependent similarity matrix based on Query-Key dot products.
    """
    def __init__(self, num_joints, d_model):
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        
        # Projection layers for data-dependent query and key
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        
        # Base learnable parameter matrix of size 100 x 100
        self.A_lrn = nn.Parameter(torch.randn(100, 100) * 0.02)
        
        # Coefficients for balancing fixed, learned and data-dependent graphs
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.zeros_(self.fc_q.bias)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.zeros_(self.fc_k.bias)

    def forward(self, x, A_fix=None):
        """
        Args:
            x: Temporal features of shape (B, T, D)
            A_fix: Fixed adjacency matrix (optional)
        Returns:
            A_final: Refined adjacency matrix of shape (B, T, T)
        """
        B, T, D = x.shape
        
        # Compute data-dependent attention over time frames: (B, T, T)
        Q = self.fc_q(x)
        K = self.fc_k(x)
        A_dep = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(D)
        
        # Handle A_fix
        if A_fix is None:
            # Construct a tridiagonal adjacency matrix of shape (T, T) for consecutive frames
            device = x.device
            A_fix_tensor = torch.eye(T, device=device)
            if T > 1:
                diag_ones = torch.ones(T - 1, device=device)
                A_fix_tensor = A_fix_tensor + torch.diag(diag_ones, diagonal=1) + torch.diag(diag_ones, diagonal=-1)
            A_fix_tensor = A_fix_tensor.unsqueeze(0)  # Shape: (1, T, T)
        else:
            if A_fix.dim() == 2:
                A_fix_tensor = A_fix.unsqueeze(0)
            else:
                A_fix_tensor = A_fix
            
            # Interpolate if shape doesn't match T
            if A_fix_tensor.shape[-1] != T:
                A_fix_tensor = F.interpolate(
                    A_fix_tensor.unsqueeze(1) if A_fix_tensor.dim() == 3 else A_fix_tensor,
                    size=(T, T),
                    mode='bilinear',
                    align_corners=False
                )
                if A_fix_tensor.dim() == 4:
                    A_fix_tensor = A_fix_tensor.squeeze(1)
        
        # Bilinearly interpolate the learnable parameter matrix if T is different from 100
        if T != 100:
            A_lrn = F.interpolate(
                self.A_lrn.unsqueeze(0).unsqueeze(0),
                size=(T, T),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # Shape: (1, T, T)
        else:
            A_lrn = self.A_lrn.unsqueeze(0)  # Shape: (1, T, T)
            
        # Linear combination: A_final = Softmax(alpha * A_fix + beta * A_lrn + gamma * A_dep, dim=-1)
        weighted_sum = self.alpha * A_fix_tensor + self.beta * A_lrn + self.gamma * A_dep
        A_final = F.softmax(weighted_sum, dim=-1)
        
        return A_final

if __name__ == '__main__':
    print("=" * 70)
    print("TEST ADAPTIVE GRAPH REFINEMENT")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    num_frames = 64
    num_joints = 27
    d_model = 256
    
    # Dummy input: (B, T, D)
    x = torch.randn(batch_size, num_frames, d_model)
    print(f"Input shape: {x.shape} (B, T, D)")
    
    # Initialize module
    refiner = AdaptiveGraphRefinement(
        num_joints=num_joints,
        d_model=d_model
    )
    
    print(f"Model parameters: {sum(p.numel() for p in refiner.parameters()):,}")
    
    # Forward pass
    output = refiner(x)
    print(f"Output shape: {output.shape} (B, T, T)")
    
    assert output.shape == (batch_size, num_frames, num_frames), "Incorrect output shape!"
    print("\nADAPTIVE GRAPH REFINEMENT TEST PASSED!")
    print("=" * 70)
