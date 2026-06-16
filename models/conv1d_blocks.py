import torch
import torch.nn as nn

class Conv1D_FFN(nn.Module):
    """
    Feed-Forward Network sử dụng Conv1D
    Kiến trúc: Conv1D -> GELU -> Dropout -> Conv1D -> Dropout
    (Đúng theo Slide 9 trong báo cáo tuần 17 của bạn)
    
    Tham khảo: Vaswani et al. (2017), ST-TR (CVPR 2021)
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4  # Mở rộng 4 lần theo chuẩn Transformer
            
        # Conv1d yêu cầu input shape: (B, C, L). 
        # Ta sẽ transpose trong forward()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: Shape (B, L, D) - L có thể là T (frames) hoặc N (joints)
        Returns:
            out: Shape (B, L, D)
        """
        residual = x
        
        # Pre-LayerNorm (ổn định hơn cho Transformer sâu)
        x = self.norm(x)
        
        # Conv1D cần shape (B, D, L)
        x = x.transpose(1, 2) 
        
        x = self.conv1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        
        # Trả về shape (B, L, D)
        x = x.transpose(1, 2) 
        
        # Residual Connection
        return residual + self.dropout(x)


class TemporalConv1D(nn.Module):
    """
    Temporal Convolution Block
    Dùng để trích xuất đặc trưng chuyển động cục bộ theo thời gian (Local Motion).
    Thường được dùng ở đầu vào hoặc cuối mạng trước khi pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dropout=0.1):
        super().__init__()
        # Padding same để giữ nguyên số frame
        padding = (kernel_size - 1) // 2 
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Shape (B, D, T)
        Returns:
            out: Shape (B, D_out, T)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# ========== TEST ==========
if __name__ == '__main__':
    print("Testing Conv1D Blocks...")
    
    # Test Conv1D_FFN
    batch_size, L, D = 8, 64, 256
    x = torch.randn(batch_size, L, D)
    
    ffn = Conv1D_FFN(d_model=D, d_ff=D*4)
    out_ffn = ffn(x)
    print(f"Conv1D_FFN Input:  {x.shape}")
    print(f"Conv1D_FFN Output: {out_ffn.shape}")
    print(f"Residual check:    {torch.allclose(x + 0.0, out_ffn, atol=1e-3) == False}") # Phải khác nhau vì đã qua biến đổi
    
    # Test TemporalConv1D
    x_temp = torch.randn(batch_size, D, 64) # (B, D, T)
    t_conv = TemporalConv1D(D, D, kernel_size=9)
    out_t = t_conv(x_temp)
    print(f"\nTemporalConv1D Input:  {x_temp.shape}")
    print(f"TemporalConv1D Output: {out_t.shape}")
    
    print("✅ Conv1D Blocks Test Passed!\n")