import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conv1d_blocks import Conv1D_FFN

class FFNBlock(nn.Module):
    """
    Feed-Forward Network Block
    Conv1D → GELU → Conv1D
    
    Tham khảo từ báo cáo tuần 17 và Vaswani et al. (2017)
    """
    
    def __init__(
        self,
        d_model=256,
        d_ff=1024,  # Thường = 4 * d_model
        dropout=0.1
    ):
        super().__init__()
        
        # Conv1D layers (thay vì Linear)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        
        # Activation
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        
        Returns:
            output: (B, T, D)
        """
        # Store for residual
        residual = x
        
        # Conv1D cần input shape (B, D, T)
        x = x.transpose(1, 2)  # (B, D, T)
        
        # First Conv1D + Activation
        x = self.conv1(x)  # (B, d_ff, T)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second Conv1D
        x = self.conv2(x)  # (B, D, T)
        
        # Transpose back to (B, T, D)
        x = x.transpose(1, 2)
        
        # Dropout
        x = self.dropout(x)
        
        # Residual connection + LayerNorm
        output = self.norm(residual + x)
        
        return output


class MHSA_Block(nn.Module):
    """
    Complete Multi-Head Self-Attention Block
    
    Kiến trúc từ báo cáo tuần 17:
    1. Graph-Augmented Multi-Head Self-Attention
    2. LayerNorm + Add (Residual)
    3. FFN Block (Conv1D → GELU → Conv1D)
    4. LayerNorm + Add (Residual)
    
    Lặp lại ×2 layers
    """
    
    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_joints=27,
        d_ff=512,
        dropout=0.1,
        graph_lambda=0.1,
        num_layers=2
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Tạo num_layers encoder layers
        self.layers = nn.ModuleList([
            MHSAEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                num_joints=num_joints,
                d_ff=d_ff,
                dropout=dropout,
                graph_lambda=graph_lambda
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
            mask: Optional mask
        
        Returns:
            output: (B, T, D)
        """
        attn_weights_list = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attn_weights_list.append(attn_weights)
        
        return x, attn_weights_list


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = torch.floor(random_tensor + keep_prob)
    return x / keep_prob * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MHSAEncoderLayer(nn.Module):
    """
    Single Encoder Layer trong MHSA Block
    
    Cấu trúc:
    1. Graph-Augmented Multi-Head Self-Attention
    2. LayerNorm + Add
    3. FFN Block (Conv1D → GELU → Conv1D)
    4. LayerNorm + Add
    """
    
    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_joints=27,
        d_ff=512,
        dropout=0.1,
        graph_lambda=0.1,
        drop_path_prob=0.0
    ):
        super().__init__()
        
        # Sub-layer 1: Graph-Augmented Multi-Head Self-Attention
        self.self_attn = GraphAugmentedAttention(
            d_model=d_model,
            nhead=nhead,
            num_joints=num_joints,
            dropout=dropout,
            graph_lambda=graph_lambda
        )
        
        # LayerNorm cho sub-layer 1
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Sub-layer 2: FFN Block
        self.ffn = Conv1D_FFN(
            d_model=d_model, 
            d_ff=d_ff, 
            dropout=dropout
        )
        # LayerNorm cho sub-layer 2
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_prob)
    
    def forward(self, x, mask=None, graph_adjacency=None):
        """
        Args:
            x: (B, T, D)
            mask: Optional mask
            graph_adjacency: Optional dynamic graph adjacency of shape (B, T, T)
        
        Returns:
            output: (B, T, D)
            attn_weights: (B, h, T, T)
        """
        # ========== SUB-LAYER 1: Self-Attention ==========
        # Pre-LN architecture (LayerNorm trước)
        x_norm = self.norm1(x)
        attn_output, attn_weights = self.self_attn(x_norm, mask, graph_adjacency)
        
        # Residual connection + Dropout + DropPath
        x = x + self.drop_path(self.dropout1(attn_output))
        
        # ========== SUB-LAYER 2: FFN ==========
        # Pre-LN architecture
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        
        # Residual connection + Dropout + DropPath
        x = x + self.drop_path(self.dropout2(ffn_output))
        
        return x, attn_weights

class GraphAugmentedAttention(nn.Module):
    """
    Multi-Head Self-Attention với các heads được thiết kế chuyên biệt (Graph-Augmented)
    """
    
    def __init__(
        self,
        d_model=128,
        nhead=4,
        num_joints=27,
        dropout=0.1,
        graph_lambda=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_joints = num_joints
        self.graph_lambda = graph_lambda
        self.d_k = d_model // nhead
        
        # ========== SHARED PROJECTIONS ==========
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # ========== GRAPH ADJACENCY ==========
        self.A_fix = self._init_skeleton_graph(num_joints)
        self.A_fix = nn.Parameter(self.A_fix, requires_grad=False)
        self.A_dyn = nn.Parameter(torch.randn(num_joints, num_joints))
        self.fc_q_graph = nn.Linear(d_model, d_model)
        self.fc_k_graph = nn.Linear(d_model, d_model)
        self.alpha = nn.Parameter(torch.tensor(0.4))
        self.beta = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.3))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        self._init_weights()
    
    def forward(self, x, mask=None, graph_adjacency=None):
        """
        Forward pass for simplified Multi-Head Attention with Graph Adjacency
        """
        B, T, D = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)  # (B, T, D)
        V = self.W_v(x)  # (B, T, D)
        
        # Reshape and transpose for multi-head: (B, h, T, d_k)
        Q = Q.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        K = K.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        V = V.view(B, T, self.nhead, self.d_k).transpose(1, 2)
        
        # Standard scaled dot-product attention: (B, h, T, T)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Add graph adjacency (shared or computed)
        if graph_adjacency is not None:
            A_final = graph_adjacency
        else:
            A_final = self._compute_graph_adjacency(x)  # (B, T, T)
        A_final_expanded = A_final.unsqueeze(1)  # (B, 1, T, T)
        
        # Add graph bias to attention scores
        attn_scores = attn_scores + self.graph_lambda * A_final_expanded
        
        # Softmax & Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, h, T, T)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Attend to values: (B, h, T, d_k)
        attn_output = torch.matmul(attn_weights, V)
        
        # Transpose and reshape back to (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        
        # Output projection
        output = self.W_o(attn_output)
        output = self.proj_dropout(output)
        
        return output, attn_weights
    
    def _compute_graph_adjacency(self, x):
        """Tính A_final = α·A_fix + β·A_dyn + γ·A_dep"""
        B, T, D = x.shape
        
        if T != self.num_joints:
            A_fix = F.interpolate(
                self.A_fix.unsqueeze(0).unsqueeze(0).float(),
                size=(T, T),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            A_dyn = F.interpolate(
                self.A_dyn.unsqueeze(0).unsqueeze(0).float(),
                size=(T, T),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        else:
            A_fix = self.A_fix
            A_dyn = self.A_dyn
        
        Q_graph = self.fc_q_graph(x)
        K_graph = self.fc_k_graph(x)
        A_dep = F.softmax(
            torch.bmm(Q_graph, K_graph.transpose(1, 2)) / math.sqrt(D),
            dim=-1
        )
        
        A_final = (
            self.alpha * A_fix.unsqueeze(0) +
            self.beta * A_dyn.unsqueeze(0) +
            self.gamma * A_dep
        )
        
        return A_final
    
    def _init_skeleton_graph(self, num_joints):
        A = torch.zeros(num_joints, num_joints)
        connections = [
            (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
            (1, 2), (2, 3), (3, 4),
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (17, 18), (18, 19), (19, 20),
        ]
        for i, j in connections:
            A[i, j] = 1
            A[j, i] = 1
        for i in range(num_joints):
            A[i, i] = 1
        return A
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
# ========== TEST MODULES ==========

if __name__ == '__main__':
    print("=" * 70)
    print("TEST MULTI-HEAD SELF-ATTENTION BLOCK")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    num_frames = 64
    num_joints = 27
    d_model = 256
    nhead = 8
    d_ff = 1024
    
    # ========== Test 1: Graph-Augmented Attention ==========
    print("\n" + "=" * 70)
    print("TEST 1: GRAPH-AUGMENTED MULTI-HEAD SELF-ATTENTION")
    print("=" * 70)
    
    x = torch.randn(batch_size, num_joints, d_model)
    print(f"\nInput shape: {x.shape} (B, N, D)")
    
    attn_module = GraphAugmentedAttention(
        d_model=d_model,
        nhead=nhead,
        num_joints=num_joints,
        dropout=0.1,
        graph_lambda=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in attn_module.parameters()):,}")
    
    output, attn_weights = attn_module(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    
    # Check graph adjacency
    print(f"\nGraph adjacency matrix:")
    print(f"  A_fix shape: {attn_module.A_fix.shape}")
    print(f"  A_dyn shape: {attn_module.A_dyn.shape}")
    print(f"  Alpha: {attn_module.alpha.item():.3f}")
    print(f"  Beta: {attn_module.beta.item():.3f}")
    print(f"  Gamma: {attn_module.gamma.item():.3f}")
    
    # ========== Test 2: FFN Block ==========
    print("\n" + "=" * 70)
    print("TEST 2: FFN BLOCK (Conv1D -> GELU -> Conv1D)")
    print("=" * 70)
    
    ffn = FFNBlock(
        d_model=d_model,
        d_ff=d_ff,
        dropout=0.1
    )
    
    print(f"\nFFN parameters: {sum(p.numel() for p in ffn.parameters()):,}")
    
    ffn_output = ffn(x)
    print(f"FFN output shape: {ffn_output.shape}")
    
    # ========== Test 3: Complete MHSA Block ==========
    print("\n" + "=" * 70)
    print("TEST 3: COMPLETE MHSA BLOCK (x2 layers)")
    print("=" * 70)
    
    mhsa_block = MHSA_Block(
        d_model=d_model,
        nhead=nhead,
        num_joints=num_joints,
        d_ff=d_ff,
        dropout=0.1,
        graph_lambda=0.1,
        num_layers=2
    )
    
    print(f"\nMHSA Block parameters: {sum(p.numel() for p in mhsa_block.parameters()):,}")
    
    output, attn_weights_list = mhsa_block(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Number of attention weight matrices: {len(attn_weights_list)}")
    print(f"Each attention weights shape: {attn_weights_list[0].shape}")
    
    # ========== Test 4: Temporal Input ==========
    print("\n" + "=" * 70)
    print("TEST 4: TEMPORAL INPUT (frames instead of joints)")
    print("=" * 70)
    
    x_temporal = torch.randn(batch_size, num_frames, d_model)
    print(f"\nInput shape: {x_temporal.shape} (B, T, D)")
    
    mhsa_temporal = MHSA_Block(
        d_model=d_model,
        nhead=nhead,
        num_joints=num_joints,  # Vẫn dùng num_joints cho graph
        d_ff=d_ff,
        dropout=0.1,
        graph_lambda=0.1,
        num_layers=2
    )
    
    output_temporal, _ = mhsa_temporal(x_temporal)
    print(f"Output shape: {output_temporal.shape}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)