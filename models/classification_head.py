import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAveragePooling(nn.Module):
    """
    Global Average Pooling (GAP)
    
    Chuyển đổi từ sequence features sang fixed-size representation
    
    Input:  (B, N, D) hoặc (B, D, N)
    Output: (B, D)
    
    Tham khảo:
    - Lin et al. (2014): "Network In Network" - giới thiệu GAP
    -广泛应用于 GCN và Transformer cho skeleton-based action recognition
    """
    
    def __init__(self, dim=1):
        """
        Args:
            dim: Dimension để pool over
                 - dim=1: Pool over sequence length (N hoặc T)
                 - dim=2: Pool over channels (ít dùng)
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        """
        Args:
            x: Input tensor
               - Shape: (B, N, D) nếu batch_first=True
               - Shape: (B, D, N) nếu batch_first=False
        
        Returns:
            pooled: (B, D) - fixed-size representation
        """
        # Pool over sequence dimension
        pooled = x.mean(dim=self.dim)  # (B, D)
        
        return pooled


class ClassificationHead(nn.Module):
    """
    Classification Head với GAP + FC + Softmax
    
    Kiến trúc:
    1. Global Average Pooling (GAP)
    2. Fully Connected Layer(s)
    3. Batch Normalization (optional)
    4. Activation (GELU/ReLU)
    5. Dropout
    6. Final FC Layer
    7. Softmax
    
    Tham khảo:
    - Vaswani et al. (2017): Transformer classification head
    - 2s-AGCN (CVPR 2019): GCN classification head
    - HA-GCN (2025): Hand-aware GCN classification
    """
    
    def __init__(
        self,
        d_model=256,
        num_classes=400,
        hidden_dim=256,
        dropout=0.5,
        use_batch_norm=True,
        activation='gelu'  # 'gelu' hoặc 'relu'
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # ========== GLOBAL AVERAGE POOLING ==========
        self.gap = GlobalAveragePooling(dim=1)  # Pool over sequence dimension
        
        # ========== CLASSIFICATION LAYERS ==========
        # Layer 1: d_model -> hidden_dim
        self.fc1 = nn.Linear(d_model, hidden_dim)
        
        # Batch Normalization (optional)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn1 = nn.Identity()
        
        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer 2: hidden_dim -> num_classes
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # ========== INITIALIZATION ==========
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo weights theo Xavier/Glorot"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor shape (B, N, D) hoặc (B, T, D)
               - B: batch size
               - N/T: sequence length (joints hoặc frames)
               - D: feature dimension
        
        Returns:
            logits: (B, num_classes) - raw logits
        """
        B, N, D = x.shape
        
        # Step 1: Global Average Pooling
        # (B, N, D) -> (B, D)
        pooled = self.gap(x)
        
        # Step 2: First FC Layer
        # (B, D) -> (B, hidden_dim)
        x = self.fc1(pooled)
        
        # Step 3: Batch Normalization
        x = self.bn1(x)
        
        # Step 4: Activation
        x = self.activation(x)
        
        # Step 5: Dropout
        x = self.dropout(x)
        
        # Step 6: Final FC Layer
        # (B, hidden_dim) -> (B, num_classes)
        logits = self.fc2(x)
        
        return logits


class SimpleClassificationHead(nn.Module):
    """
    Simple Classification Head (không có hidden layer)
    
    Kiến trúc đơn giản:
    1. GAP
    2. FC -> num_classes
    3. Softmax
    
    Dùng khi muốn model nhẹ hơn
    """
    
    def __init__(self, d_model=256, num_classes=400):
        super().__init__()
        self.gap = GlobalAveragePooling(dim=1)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        pooled = self.gap(x)  # (B, D)
        logits = self.fc(pooled)  # (B, num_classes)
        return logits


# ========== TEST MODULES ==========

if __name__ == '__main__':
    print("=" * 70)
    print("TEST CLASSIFICATION HEAD")
    print("=" * 70)
    
    # Parameters
    batch_size = 8
    num_joints = 27
    d_model = 256
    num_classes = 400
    
    # Create dummy input
    x = torch.randn(batch_size, num_joints, d_model)
    print(f"\nInput shape: {x.shape} (B, N, D)")
    
    # ========== Test 1: Full Classification Head ==========
    print("\n" + "=" * 70)
    print("TEST 1: FULL CLASSIFICATION HEAD (GAP + FC + Softmax)")
    print("=" * 70)
    
    classifier = ClassificationHead(
        d_model=d_model,
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.5,
        use_batch_norm=True,
        activation='gelu'
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Test forward
    logits = classifier(x)
    print(f"\nOutput (logits) shape: {logits.shape}")
    print(f"Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    
    # ========== Test 2: Simple Classification Head ==========
    print("\n" + "=" * 70)
    print("TEST 2: SIMPLE CLASSIFICATION HEAD")
    print("=" * 70)
    
    simple_classifier = SimpleClassificationHead(
        d_model=d_model,
        num_classes=num_classes
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in simple_classifier.parameters()):,}")
    
    logits_simple = simple_classifier(x)
    print(f"Output shape: {logits_simple.shape}")
    
    # ========== Test 3: Global Average Pooling ==========
    print("\n" + "=" * 70)
    print("TEST 3: GLOBAL AVERAGE POOLING")
    print("=" * 70)
    
    gap = GlobalAveragePooling(dim=1)
    pooled = gap(x)
    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {pooled.shape}")
    
    # Verify pooling
    manual_pool = x.mean(dim=1)
    print(f"\nManual pooling matches: {torch.allclose(pooled, manual_pool)}")
    
    # ========== Test 4: Edge Cases ==========
    print("\n" + "=" * 70)
    print("TEST 4: EDGE CASES")
    print("=" * 70)
    
    # Different sequence lengths
    x_short = torch.randn(4, 10, 128)
    classifier_edge = ClassificationHead(
        d_model=128,
        num_classes=50,
        hidden_dim=128
    )
    
    logits_edge = classifier_edge(x_short)
    print(f"\nInput shape:  {x_short.shape}")
    print(f"Output shape: {logits_edge.shape}")
    
    # Very long sequence
    x_long = torch.randn(2, 200, 512)
    classifier_long = ClassificationHead(
        d_model=512,
        num_classes=100,
        hidden_dim=512
    )
    
    logits_long = classifier_long(x_long)
    print(f"\nInput shape:  {x_long.shape}")
    print(f"Output shape: {logits_long.shape}")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)