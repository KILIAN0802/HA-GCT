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
        
        # 2. SPATIAL BRANCH: HA-GC Blocks (x3) with Stochastic Depth (DropPath)
        total_layers = num_ha_gc_blocks + num_mhsa_layers
        dpr = [0.1 + (0.3 - 0.1) * i / max(1, total_layers - 1) for i in range(total_layers)]
        
        self.spatial_branch = nn.ModuleList([
            HA_GC_Block(d_model, d_model, num_joints, drop_path_prob=dpr[i])
            for i in range(num_ha_gc_blocks)
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
                graph_lambda=graph_lambda,
                drop_path_prob=dpr[num_ha_gc_blocks + i]
            )
            for i in range(num_mhsa_layers)
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
    
    def forward(self, x, return_embedding=False):
        """
        Input: x shape (B, C, T, N) = (B, 2, 64, 27)
        Output: (B, num_classes) or (B, N, D)
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
        
        if return_embedding:
            return x_fused
            
        # STEP 5: Classification (GAP + Softmax)
        output = self.classifier(x_fused)  # (B, num_classes)
        
        return output

class MultiStreamHA_GCT(nn.Module):
    """
    Multi-Stream HA-GCT (Late Fusion of Joint, Bone, and Velocity)
    """
    def __init__(
        self,
        num_joints=27,
        in_channels=2,
        d_model=128,
        num_ha_gc_blocks=3,
        num_mhsa_layers=2,
        nhead=8,
        num_classes=400,
        dropout=0.5,
        graph_lambda=0.1,
        max_frames=100
    ):
        super().__init__()
        
        # Stream 1: Joint
        self.stream_joint = HA_GCT(
            num_joints=num_joints,
            in_channels=in_channels,
            d_model=d_model,
            num_ha_gc_blocks=num_ha_gc_blocks,
            num_mhsa_layers=num_mhsa_layers,
            nhead=nhead,
            num_classes=num_classes,
            dropout=dropout,
            graph_lambda=graph_lambda,
            max_frames=max_frames
        )
        
        # Stream 2: Bone
        self.stream_bone = HA_GCT(
            num_joints=num_joints,
            in_channels=in_channels,
            d_model=d_model,
            num_ha_gc_blocks=num_ha_gc_blocks,
            num_mhsa_layers=num_mhsa_layers,
            nhead=nhead,
            num_classes=num_classes,
            dropout=dropout,
            graph_lambda=graph_lambda,
            max_frames=max_frames
        )
        
        # Stream 3: Velocity
        self.stream_velocity = HA_GCT(
            num_joints=num_joints,
            in_channels=in_channels,
            d_model=d_model,
            num_ha_gc_blocks=num_ha_gc_blocks,
            num_mhsa_layers=num_mhsa_layers,
            nhead=nhead,
            num_classes=num_classes,
            dropout=dropout,
            graph_lambda=graph_lambda,
            max_frames=max_frames
        )
        
        # Skeleton topology mapping for bone calculation
        self.parents = {
            0: None,   # Nose / Root
            1: 0,      # Shoulder L
            2: 0,      # Shoulder R
            3: 1,      # Elbow L
            4: 2,      # Elbow R
            5: 3,      # Wrist L
            6: 4,      # Wrist R
            7: 5,      # Palm L
            17: 6,     # Palm R
            8: 7,      # Thumb L
            9: 7,      # Index L root
            10: 9,     # Index L tip
            11: 7,     # Middle L root
            12: 11,    # Middle L tip
            13: 7,     # Ring L root
            14: 13,    # Ring L tip
            15: 7,     # Pinky L root
            16: 15,    # Pinky L tip
            18: 17,    # Thumb R
            19: 17,    # Index R root
            20: 19,    # Index R tip
            21: 17,    # Middle R root
            22: 21,    # Middle R tip
            23: 17,    # Ring R root
            24: 23,    # Ring R tip
            25: 17,    # Pinky R root
            26: 25     # Pinky R tip
        }
        
        # Late Fusion Classification Head
        self.classifier = SimpleClassificationHead(
            d_model=3 * d_model,
            num_classes=num_classes
        )
        
    def _compute_bone(self, joint):
        # joint shape: (B, C, T, V)
        B, C, T, V = joint.shape
        bone = torch.zeros_like(joint)
        for child, parent in self.parents.items():
            if parent is not None:
                bone[:, :, :, child] = joint[:, :, :, child] - joint[:, :, :, parent]
        return bone

    def _compute_velocity(self, joint):
        # joint shape: (B, C, T, V)
        B, C, T, V = joint.shape
        velocity = torch.zeros_like(joint)
        velocity[:, :, 1:, :] = joint[:, :, 1:, :] - joint[:, :, :-1, :]
        return velocity

    def forward(self, joint):
        # Compute streams dynamically on target device
        bone = self._compute_bone(joint)
        velocity = self._compute_velocity(joint)
        
        # Get embeddings from the streams
        feat_joint = self.stream_joint(joint, return_embedding=True)
        feat_bone = self.stream_bone(bone, return_embedding=True)
        feat_velocity = self.stream_velocity(velocity, return_embedding=True)
        
        # Concatenate features
        feat_fused = torch.cat([feat_joint, feat_bone, feat_velocity], dim=-1)
        
        # Predict logits
        output = self.classifier(feat_fused)
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
    
    # Test Multi-Stream
    print("\n" + "=" * 70)
    print("TESTING MULTI-STREAM HA-GCT (LATE FUSION)")
    print("=" * 70)
    
    ms_model = MultiStreamHA_GCT(
        num_joints=num_joints,
        in_channels=in_channels,
        d_model=128,
        num_ha_gc_blocks=3,
        num_mhsa_layers=2,
        nhead=8,
        num_classes=num_classes,
        dropout=0.5,
        graph_lambda=0.1,
        max_frames=num_frames
    )
    
    print(f"Multi-Stream HA-GCT Parameters: {sum(p.numel() for p in ms_model.parameters()):,}")
    ms_output = ms_model(x)
    print(f"Multi-Stream Output shape: {ms_output.shape}")
    assert ms_output.shape == (batch_size, num_classes), "Incorrect Multi-Stream output shape!"
    print("\nMULTI-STREAM HA-GCT TEST PASSED!")
    print("=" * 70)

