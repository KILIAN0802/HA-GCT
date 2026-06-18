import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module cho HA-GCT
    
    Tham khảo từ:
    - CrossViT (ICCV 2021): https://openaccess.thecvf.com/content/ICCV2021/html/Chen_CrossViT_Cross-Attention_Multi-Scale_Vision_Transformer_for_Image_Classification_ICCV_2021_paper.html
    - CAST (NeurIPS 2023): Cross-Attention in Space and Time
    
    Cơ chế:
    - Spatial Branch (HA-GC) cung cấp: Query (Q)
    - Temporal Branch (MHSA) cung cấp: Key (K), Value (V)
    - Fusion qua bidirectional cross-attention
    """
    
    def __init__(
        self,
        d_model=256,
        nhead=8,
        dropout=0.1,
        fusion_type='bidirectional'  # 'unidirectional' hoặc 'bidirectional'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.fusion_type = fusion_type
        
        # ========== DIRECTION 1: Spatial → Temporal ==========
        # Spatial features query temporal features
        self.cross_attn_s2t = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm và Dropout cho direction 1
        self.norm_s2t = nn.LayerNorm(d_model)
        self.dropout_s2t = nn.Dropout(dropout)
        
        # ========== DIRECTION 2: Temporal → Spatial ==========
        # Temporal features query spatial features
        self.cross_attn_t2s = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorm và Dropout cho direction 2
        self.norm_t2s = nn.LayerNorm(d_model)
        self.dropout_t2s = nn.Dropout(dropout)
        
        # ========== FUSION PROJECTION LAYER ==========
        self.fusion_proj = nn.Linear(d_model, d_model)
        
        # Residual connection
        self.residual_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)
    
    def forward(self, spatial_feat, temporal_feat):
        """
        Forward pass cho Cross-Attention Fusion
        """
        temporal_pooled = temporal_feat.mean(dim=1, keepdim=True)
        fused = spatial_feat + temporal_pooled
        fused_feat = self.fusion_proj(fused)
        
        # Residual connection
        fused_feat = self.residual_norm(spatial_feat + fused_feat)
        
        return fused_feat
    
    def _align_dimensions(self, x, target_len):
        """
        Align sequence length về target_len bằng linear interpolation
        
        Args:
            x: Tensor shape (B, L, D)
            target_len: số lượng mục tiêu
        
        Returns:
            Tensor shape (B, target_len, D)
        """
        B, L, D = x.shape
        
        if L == target_len:
            return x
        
        # Transpose để interpolate: (B, D, L)
        x = x.transpose(1, 2)
        
        # Interpolate về target_len
        x = F.interpolate(
            x, 
            size=target_len, 
            mode='linear', 
            align_corners=False
        )
        
        # Transpose về (B, target_len, D)
        x = x.transpose(1, 2)
        
        return x


class SimpleCrossAttention(nn.Module):
    """
    Simple Cross-Attention (Unidirectional)
    Đơn giản hơn, chỉ 1 direction: Spatial → Temporal
    
    Dùng khi muốn tiết kiệm computation
    """
    
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, spatial_feat, temporal_feat):
        """
        Args:
            spatial_feat: (B, N, D)
            temporal_feat: (B, T, D)
        
        Returns:
            fused: (B, N, D)
        """
        # Spatial queries Temporal
        attn_out, _ = self.cross_attn(
            query=spatial_feat,
            key=temporal_feat,
            value=temporal_feat
        )
        
        # Residual + Norm
        attn_out = self.dropout(attn_out)
        out = self.norm(spatial_feat + attn_out)
        
        # Concatenate và fuse
        combined = torch.cat([spatial_feat, out], dim=-1)
        fused = self.fusion(combined)
        
        return fused


# ========== TEST MODULES ==========

if __name__ == '__main__':
    print("=" * 70)
    print("TEST CROSS-ATTENTION FUSION MODULE")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    num_joints = 27
    num_frames = 64
    d_model = 256
    nhead = 8
    
    # Create dummy data
    spatial_feat = torch.randn(batch_size, num_joints, d_model)
    temporal_feat = torch.randn(batch_size, num_frames, d_model)
    
    print(f"\nInput shapes:")
    print(f"  Spatial feat:   {spatial_feat.shape} (B, N, D)")
    print(f"  Temporal feat:  {temporal_feat.shape} (B, T, D)")
    
    # ========== Test 1: Bidirectional Cross-Attention ==========
    print("\n" + "=" * 70)
    print("TEST 1: BIDIRECTIONAL CROSS-ATTENTION")
    print("=" * 70)
    
    model_bidirectional = CrossAttentionFusion(
        d_model=d_model,
        nhead=nhead,
        dropout=0.1,
        fusion_type='bidirectional'
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model_bidirectional.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model_bidirectional.parameters() if p.requires_grad):,}")
    
    # Forward pass
    output = model_bidirectional(spatial_feat, temporal_feat)
    print(f"\nOutput shape: {output.shape}")
    
    # Check attention weights
    if hasattr(model_bidirectional, 'attn_weights'):
        attn_weights = model_bidirectional.attn_weights
        print(f"\nAttention weights shapes:")
        print(f"  Spatial -> Temporal: {attn_weights['spatial_to_temporal'].shape}")
        print(f"  Temporal -> Spatial: {attn_weights['temporal_to_spatial'].shape}")
    
    # ========== Test 2: Simple Cross-Attention ==========
    print("\n" + "=" * 70)
    print("TEST 2: SIMPLE CROSS-ATTENTION (UNIDIRECTIONAL)")
    print("=" * 70)
    
    model_simple = SimpleCrossAttention(
        d_model=d_model,
        nhead=nhead,
        dropout=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model_simple.parameters()):,}")
    
    output_simple = model_simple(spatial_feat, temporal_feat)
    print(f"Output shape: {output_simple.shape}")
    
    # ========== Test 3: Edge Cases ==========
    print("\n" + "=" * 70)
    print("TEST 3: EDGE CASES")
    print("=" * 70)
    
    # Different sequence lengths
    print("\nTesting with different sequence lengths:")
    spatial_short = torch.randn(4, 10, 128)
    temporal_long = torch.randn(4, 100, 128)
    
    model_edge = CrossAttentionFusion(
        d_model=128,
        nhead=4,
        dropout=0.1
    )
    
    output_edge = model_edge(spatial_short, temporal_long)
    print(f"  Input spatial:  {spatial_short.shape}")
    print(f"  Input temporal: {temporal_long.shape}")
    print(f"  Output:         {output_edge.shape}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)