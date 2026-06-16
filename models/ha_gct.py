import sys
sys.path.extend(['./', '../', 'models'])

import torch
import torch.nn as nn

from models.physical_embedding import PhysicalEmbedding
from models.ha_gc_block import HA_GC_Block
from models.mhsa import MHSAEncoderLayer
from models.adaptive_graph import AdaptiveGraphRefinement
from models.cross_attention import CrossAttentionFusion
from models.classification_head import SimpleClassificationHead

class HA_GCT(nn.Module):
    """
    HA-GCT: Hand-Aware Graph Spatio-Temporal Transformer Network
    
    Complete unified model based on the architectural diagram:
    1. Physical Embedding + Positional Encoding (TPE + SPE)
    2. Spatial Branch: HA-GC Block x3 (Hand-Aware Graph Convolutions)
    3. Temporal Branch: Graph-Augmented MHSA Encoder x2
    4. Adaptive Graph Refinement
    5. Bidirectional Cross-Attention Fusion
    6. Classification Head (GAP + Softmax)
    """
    def __init__(
        self,
        num_joints=27,
        in_channels=2,
        d_model=256,
        num_ha_gc_blocks=3,      # x3 as in diagram
        num_mhsa_layers=2,       # x2 as in diagram
        nhead=8,
        num_classes=400,         # 400VSL dataset
        dropout=0.1,
        graph_lambda=0.1,        # lambda in MHSA formula
        max_frames=100
    ):
        super().__init__()
        
        # 1. INPUT EMBEDDING + PE
        self.physical_embedding = PhysicalEmbedding(
            in_channels=in_channels,
            num_joints=num_joints,
            d_model=d_model,
            max_frames=max_frames,
            dropout=dropout
        )
        
        # 2. SPATIAL BRANCH: HA-GC Blocks (x3)
        self.spatial_branch = nn.ModuleList([
            HA_GC_Block(d_model, d_model, num_joints)
            for _ in range(num_ha_gc_blocks)
        ])
        
        # 3. TEMPORAL BRANCH: MHSA Blocks (x2)
        # Note: MHSA here is Graph-Augmented MHSA
        self.temporal_branch = nn.ModuleList([
            MHSAEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                num_joints=num_joints,
                d_ff=d_model * 4,
                dropout=dropout,
                graph_lambda=graph_lambda
            )
            for _ in range(num_mhsa_layers)
        ])
        
        # 4. ADAPTIVE GRAPH REFINEMENT
        self.adaptive_graph = AdaptiveGraphRefinement(
            num_joints=num_joints,
            d_model=d_model
        )
        
        # 5. CROSS-ATTENTION FUSION
        self.cross_fusion = CrossAttentionFusion(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout
        )
        
        # 6. CLASSIFICATION HEAD (GAP + Softmax)
        self.classifier = SimpleClassificationHead(
            d_model=d_model,
            num_classes=num_classes
        )
    
    def forward(self, x):
        """
        Input: x shape (B, C, T, N) = (B, 2, 64, 27)
        Output: (B, num_classes)
        """
        B, C, T, N = x.shape
        
        # STEP 1: Physical Embedding
        # Transpose (B, C, T, N) -> (B, T, N, C)
        x_embed = x.permute(0, 2, 3, 1).contiguous()
        x_embed = self.physical_embedding(x_embed)  # (B, T, N, D)
        
        # STEP 2: Spatial Branch (HA-GC x3)
        # HA-GC expects shape (B, D, T, N)
        x_spatial = x_embed.permute(0, 3, 1, 2).contiguous()  # (B, D, T, N)
        
        for ha_gc_block in self.spatial_branch:
            x_spatial = ha_gc_block(x_spatial)
        
        # Pool over time: (B, D, T, N) -> (B, D, N)
        x_spatial = x_spatial.mean(dim=2)
        # Transpose: (B, D, N) -> (B, N, D)
        x_spatial = x_spatial.transpose(1, 2)
        
        # STEP 3: Temporal Branch (MHSA x2)
        # Pool over joints to extract temporal features: (B, T, N, D) -> (B, T, D)
        x_temporal = x_embed.mean(dim=2)  # (B, T, D)
        
        # Compute Adaptive Graph Adjacency
        A_final = self.adaptive_graph(x_temporal)  # (B, T, T)
        
        # MHSA with Graph-Augmented Attention
        for mhsa_layer in self.temporal_branch:
            x_temporal, attn_weights = mhsa_layer(
                x_temporal, 
                graph_adjacency=A_final
            )
        
        # STEP 4: Cross-Attention Fusion
        # Fusion between spatial (x_spatial) and temporal (x_temporal)
        x_fused = self.cross_fusion(x_spatial, x_temporal)  # (B, N, D)
        
        # STEP 5: Classification (GAP + Softmax)
        output = self.classifier(x_fused)  # (B, num_classes)
        
        return output

if __name__ == '__main__':
    print("=" * 70)
    print("TESTING FULL UNIFIED HA-GCT NETWORK")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    in_channels = 2
    num_frames = 64
    num_joints = 27
    num_classes = 400
    
    # Dummy input: (B, C, T, N) = (8, 2, 64, 27)
    x = torch.randn(batch_size, in_channels, num_frames, num_joints)
    print(f"Input shape: {x.shape} (B, C, T, N)")
    
    # Initialize HA-GCT model
    model = HA_GCT(
        num_joints=num_joints,
        in_channels=in_channels,
        d_model=256,
        num_ha_gc_blocks=3,
        num_mhsa_layers=2,
        nhead=8,
        num_classes=num_classes,
        dropout=0.1,
        graph_lambda=0.1
    )
    
    print(f"\nHA-GCT Network Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass
    output = model(x)
    print(f"Output logits shape: {output.shape} (B, num_classes)")
    
    assert output.shape == (batch_size, num_classes), "Incorrect output shape!"
    print("\nHA-GCT FULL NETWORK TEST PASSED!")
    print("=" * 70)
