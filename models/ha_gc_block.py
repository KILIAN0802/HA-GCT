import sys
sys.path.extend(['./', '../'])

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from graph.sign_27 import Graph as SpatialGraph
from graph.sign_27_A_hands import Graph as HandsGraph

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor

def masked_row_normalize(A, mask, eps=1e-6):
    A = A.masked_fill(mask == 0, 0.0)
    row_sum = A.sum(dim=-1, keepdim=True).clamp_min(eps)
    return A / row_sum

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class HA_GC_Block(nn.Module):
    """
    HA-GC Block (Hand-Aware Graph Convolution Block)
    
    Performs hand-aware graph convolutions over skeletal joints.
    Input: (B, C_in, T, V)
    Output: (B, C_out, T, V)
    """
    def __init__(self, in_channels, out_channels, num_joints=27, num_subset=3, drop_path_prob=0.0):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_subset = num_subset
        
        # Load spatial skeleton graph and hands graph
        self.spatial_graph = SpatialGraph(labeling_mode='spatial')
        self.hands_graph = HandsGraph(labeling_mode='spatial')
        
        A = self.spatial_graph.A  # (3, 27, 27)
        A_hands = self.hands_graph.A  # (3, 27, 27)
        
        # Define parameters matching unit_gcn in hand_aware_sl_lgcn.py
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.register_buffer('A_hands', torch.tensor(A_hands, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        
        self.PA_hands = nn.Parameter(torch.from_numpy(A_hands.astype(np.float32)))
        self.beta = nn.Parameter(torch.tensor([0.5], dtype=torch.float32))
        
        # Conv branch initializations
        self.conv = nn.ModuleList()
        for i in range(num_subset):
            self.conv.append(nn.Conv2d(in_channels, out_channels, 1))
            
        # Residual branch
        if in_channels != out_channels:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res = nn.Identity()
            
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path_prob)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.bn.weight, 1e-6)

    def forward(self, x):
        """
        Args:
            x: Input tensor shape (B, C_in, T, V)
        Returns:
            out: Output tensor shape (B, C_out, T, V)
        """
        B, C, T, V = x.size()
        
        # Adjacency matrices are registered buffers/parameters, so they are already on the correct device
        A = self.A
        A_hands = self.A_hands
        PA_hands = self.PA_hands
        
        A_base_raw = torch.relu(self.A + self.PA)
        A_base_mask = (self.A > 0).float()
        A_base = masked_row_normalize(A_base_raw, A_base_mask)
        
        A_hand_raw = torch.relu(self.A_hands * self.alpha + self.PA_hands * self.beta)
        A_hand_mask = (self.A_hands > 0).float()
        A_hand = masked_row_normalize(A_hand_raw, A_hand_mask)
        
        A = A_base + A_hand
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        
        y = None
        for i in range(self.num_subset):
            f = self.conv[i](x)  # (B, C_out, T, V)
            
            # Matrix multiplication strictly over spatial dimension (V) using einsum
            z = torch.einsum('bctv,vw->bctw', f, A[i])
            y = z + y if y is not None else z
            
        y = self.bn(y)
        y = self.res(x) + self.drop_path(y)
        
        return self.relu(y)

if __name__ == '__main__':
    print("=" * 70)
    print("TEST HA-GC BLOCK")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    channels = 256
    num_frames = 64
    num_joints = 27
    
    # Dummy input: (B, C, T, V)
    x = torch.randn(batch_size, channels, num_frames, num_joints)
    print(f"Input shape: {x.shape} (B, C, T, V)")
    
    # Initialize block
    block = HA_GC_Block(
        in_channels=channels,
        out_channels=channels,
        num_joints=num_joints
    )
    
    print(f"Model parameters: {sum(p.numel() for p in block.parameters()):,}")
    
    # Forward pass
    output = block(x)
    print(f"Output shape: {output.shape} (B, C_out, T, V)")
    
    assert output.shape == (batch_size, channels, num_frames, num_joints), "Incorrect output shape!"
    print("\nHA-GC BLOCK TEST PASSED!")
    print("=" * 70)
