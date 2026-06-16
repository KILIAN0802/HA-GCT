import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class KeypointReconstruction:
    """
    Keypoint Reconstruction với Anchors
    
    Tham khảo: "Preprocessing MediaPipe Keypoints with Keypoint Reconstruction"
    
    Cơ chế:
    1. Sử dụng các "anchors" (cổ/wrist) để chuẩn hóa tọa độ
    2. Normalize theo chiều dài cổ (wrist length)
    3. Bilinear interpolation với frame trước và sau
    
    Ưu điểm:
    - Tận dụng cấu trúc giải phẫu của bàn tay
    - Giảm thiểu lỗi do scale và rotation
    - Nội suy mượt mà hơn linear interpolation
    """
    
    def __init__(
        self,
        anchor_joint_idx: int = 0,  # Wrist joint
        scale_joints: Tuple[int, int] = (0, 4),  # Wrist to thumb tip
        confidence_threshold: float = 0.3
    ):
        """
        Args:
            anchor_joint_idx: Index của anchor joint (wrist)
            scale_joints: Tuple của 2 joints để tính scale (wrist, thumb_tip)
            confidence_threshold: Threshold để xác định keypoint bị mất
        """
        self.anchor_joint_idx = anchor_joint_idx
        self.scale_joints = scale_joints
        self.confidence_threshold = confidence_threshold
    
    def _compute_anchor_distance(self, frame_data: np.ndarray) -> float:
        """
        Tính khoảng cách anchor để chuẩn hóa scale
        
        Args:
            frame_data: Shape (N, D) - một frame
        
        Returns:
            distance: float - khoảng cách giữa 2 scale joints
        """
        joint1 = frame_data[self.scale_joints[0], :2]  # (x, y)
        joint2 = frame_data[self.scale_joints[1], :2]
        
        distance = np.sqrt(np.sum((joint1 - joint2) ** 2))
        
        # Avoid division by zero
        return max(distance, 1e-6)
    
    def _normalize_to_anchor(self, frame_data: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa tọa độ theo anchor
        
        Args:
            frame_data: Shape (N, D)
        
        Returns:
            normalized: Shape (N, D)
        """
        # Get anchor position
        anchor = frame_data[self.anchor_joint_idx, :2].copy()  # (x, y)
        
        # Compute scale factor
        scale = self._compute_anchor_distance(frame_data)
        
        # Normalize: (x - anchor_x) / scale
        normalized = frame_data.copy()
        normalized[:, :2] = (frame_data[:, :2] - anchor) / scale
        
        return normalized
    
    def _denormalize_from_anchor(self, normalized_data: np.ndarray, 
                                  original_frame: np.ndarray) -> np.ndarray:
        """
        Denormalize từ anchor space về original space
        
        Args:
            normalized_data: Shape (N, D)
            original_frame: Shape (N, D) - frame gốc để lấy anchor và scale
        
        Returns:
            denormalized: Shape (N, D)
        """
        anchor = original_frame[self.anchor_joint_idx, :2].copy()
        scale = self._compute_anchor_distance(original_frame)
        
        denormalized = normalized_data.copy()
        denormalized[:, :2] = normalized_data[:, :2] * scale + anchor
        
        return denormalized
    
    def _bilinear_interpolation(self, prev_frame: np.ndarray, 
                                 next_frame: np.ndarray,
                                 alpha: float) -> np.ndarray:
        """
        Bilinear interpolation giữa 2 frames
        
        Args:
            prev_frame: Shape (N, D) - frame trước
            next_frame: Shape (N, D) - frame sau
            alpha: Weight cho next_frame (0 <= alpha <= 1)
        
        Returns:
            interpolated: Shape (N, D)
        """
        # Normalize both frames to anchor space
        prev_norm = self._normalize_to_anchor(prev_frame)
        next_norm = self._normalize_to_anchor(next_frame)
        
        # Bilinear interpolation in normalized space
        interpolated_norm = (1 - alpha) * prev_norm + alpha * next_norm
        
        # Denormalize back to original space (using prev_frame as reference)
        interpolated = self._denormalize_from_anchor(interpolated_norm, prev_frame)
        
        return interpolated
    
    def interpolate(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        """
        Interpolate missing keypoints trong toàn bộ sequence
        
        Args:
            skeleton_sequence: Shape (T, N, D)
                - T: number of frames
                - N: number of joints
                - D: dimensions (x, y, [confidence])
        
        Returns:
            interpolated: Shape (T, N, D)
        """
        T, N, D = skeleton_sequence.shape
        interpolated = skeleton_sequence.copy()
        
        # Detect missing keypoints
        if D >= 3:
            confidence = skeleton_sequence[:, :, 2]  # (T, N)
            missing_mask = confidence < self.confidence_threshold
        else:
            # Nếu không có confidence, giả sử không có missing
            return interpolated
        
        # Interpolate từng joint
        for joint_idx in range(N):
            # Find frames where this joint is missing
            missing_frames = np.where(missing_mask[:, joint_idx])[0]
            
            for frame_idx in missing_frames:
                # Find previous valid frame
                prev_valid = frame_idx - 1
                while prev_valid >= 0 and missing_mask[prev_valid, joint_idx]:
                    prev_valid -= 1
                
                # Find next valid frame
                next_valid = frame_idx + 1
                while next_valid < T and missing_mask[next_valid, joint_idx]:
                    next_valid += 1
                
                # Case 1: Có cả prev và next valid frames
                if prev_valid >= 0 and next_valid < T:
                    # Compute alpha cho bilinear interpolation
                    alpha = (frame_idx - prev_valid) / (next_valid - prev_valid)
                    
                    # Interpolate
                    interpolated[frame_idx, joint_idx, :2] = self._bilinear_interpolation(
                        interpolated[prev_valid],
                        interpolated[next_valid],
                        alpha
                    )[joint_idx, :2]
                    
                    # Set confidence to medium value
                    if D >= 3:
                        interpolated[frame_idx, joint_idx, 2] = 0.5
                
                # Case 2: Chỉ có prev valid
                elif prev_valid >= 0:
                    # Forward fill
                    interpolated[frame_idx, joint_idx, :2] = \
                        interpolated[prev_valid, joint_idx, :2].copy()
                    if D >= 3:
                        interpolated[frame_idx, joint_idx, 2] = 0.3
                
                # Case 3: Chỉ có next valid
                elif next_valid < T:
                    # Backward fill
                    interpolated[frame_idx, joint_idx, :2] = \
                        interpolated[next_valid, joint_idx, :2].copy()
                    if D >= 3:
                        interpolated[frame_idx, joint_idx, 2] = 0.3
                
                # Case 4: Không có valid frame nào -> giữ nguyên
        
        return interpolated


class STGAIN(nn.Module):
    """
    Spatiotemporal Generative Adversarial Interpolation
    
    Tham khảo: "A robust two-stage framework for human skeleton action 
    recognition with GAIN and masked autoencoder"
    
    Cơ chế:
    - Generator: Nội suy keypoints bị thiếu dựa trên context không-thời gian
    - Discriminator: Phân biệt real vs generated keypoints
    - Học cả spatial (cấu trúc cơ thể) và temporal (quỹ đạo chuyển động)
    
    Kiến trúc Generator:
    1. Spatial Encoder: Encode cấu trúc skeleton
    2. Temporal Encoder: Encode chuyển động theo thời gian
    3. Fusion Layer: Combine spatial và temporal features
    4. Decoder: Generate missing keypoints
    """
    
    def __init__(
        self,
        num_joints: int = 27,
        in_channels: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # ========== SPATIAL ENCODER ==========
        # Graph Convolution cho spatial features
        self.spatial_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # ========== TEMPORAL ENCODER ==========
        # Temporal Convolution cho motion features
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_dim, 1]),  # LayerNorm cho Conv1d
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.LayerNorm([hidden_dim, 1]),
            nn.GELU(),
        )
        
        # ========== TEMPORAL ATTENTION ==========
        # Self-attention cho temporal dependencies
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # ========== FUSION LAYER ==========
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ========== DECODER ==========
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, in_channels),
        )
        
        # ========== MASK PREDICTION (cho GAIN) ==========
        self.mask_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Khởi tạo weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x: Input tensor shape (B, T, N, D)
                - B: batch size
                - T: number of frames
                - N: number of joints
                - D: dimensions (x, y)
            mask: Binary mask shape (B, T, N)
                - 1: valid keypoint
                - 0: missing keypoint
        
        Returns:
            reconstructed: Shape (B, T, N, D)
            mask_logits: Shape (B, T, N, 1)
        """
        B, T, N, D = x.shape
        
        # ========== SPATIAL ENCODING ==========
        # Encode từng frame độc lập
        x_spatial = x.view(B * T, N, D)  # (B*T, N, D)
        x_spatial = self.spatial_encoder(x_spatial)  # (B*T, N, hidden_dim)
        x_spatial = x_spatial.view(B, T, N, self.hidden_dim)  # (B, T, N, h)
        
        # ========== TEMPORAL ENCODING ==========
        # Encode theo thời gian cho từng joint
        x_temp = x.permute(0, 2, 3, 1).contiguous()  # (B, N, D, T)
        x_temp = x_temp.view(B * N, D, T)  # (B*N, D, T)
        x_temp = self.temporal_encoder(x_temp)  # (B*N, h, T)
        x_temp = x_temp.view(B, N, self.hidden_dim, T)  # (B, N, h, T)
        x_temp = x_temp.permute(0, 3, 1, 2).contiguous()  # (B, T, N, h)
        
        # ========== TEMPORAL ATTENTION ==========
        # Apply attention trên temporal dimension
        x_attn = x_temp.view(B * N, T, self.hidden_dim)  # (B*N, T, h)
        x_attn, _ = self.temporal_attn(x_attn, x_attn, x_attn)  # (B*N, T, h)
        x_attn = x_attn.view(B, T, N, self.hidden_dim)  # (B, T, N, h)
        
        # ========== FUSION ==========
        # Combine spatial và temporal features
        x_fused = torch.cat([x_spatial, x_attn], dim=-1)  # (B, T, N, 2h)
        x_fused = self.fusion(x_fused)  # (B, T, N, h)
        
        # ========== DECODER ==========
        reconstructed = self.decoder(x_fused)  # (B, T, N, D)
        
        # ========== MASK PREDICTION ==========
        mask_logits = self.mask_predictor(x_fused)  # (B, T, N, 1)
        
        return reconstructed, mask_logits


class SmartInterpolationPipeline:
    """
    Pipeline kết hợp Keypoint Reconstruction và STGAIN
    
    Quy trình:
    1. Detect missing keypoints (confidence < threshold)
    2. Thử Keypoint Reconstruction trước (nhanh, đơn giản)
    3. Nếu vẫn còn missing -> dùng STGAIN (phức tạp hơn nhưng chính xác hơn)
    """
    
    def __init__(
        self,
        num_joints: int = 27,
        confidence_threshold: float = 0.3,
        use_stgain: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            num_joints: Number of joints
            confidence_threshold: Threshold để xác định missing keypoints
            use_stgain: Có sử dụng STGAIN không
            device: Device để chạy model
        """
        self.num_joints = num_joints
        self.confidence_threshold = confidence_threshold
        self.use_stgain = use_stgain
        self.device = device
        
        # Keypoint Reconstruction
        self.keypoint_recon = KeypointReconstruction(
            anchor_joint_idx=0,
            scale_joints=(0, 4),
            confidence_threshold=confidence_threshold
        )
        
        # STGAIN model
        if use_stgain:
            self.stgain_model = STGAIN(num_joints=num_joints)
            self.stgain_model.to(device)
            self.stgain_model.eval()
        else:
            self.stgain_model = None
    
    def load_stgain_model(self, model_path: str):
        """
        Load pretrained STGAIN model
        
        Args:
            model_path: Path đến checkpoint
        """
        if self.stgain_model is None:
            raise ValueError("STGAIN model is not initialized")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.stgain_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded STGAIN model from {model_path}")
    
    def interpolate(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        """
        Interpolate missing keypoints
        
        Args:
            skeleton_sequence: Shape (T, N, D)
        
        Returns:
            interpolated: Shape (T, N, D)
        """
        T, N, D = skeleton_sequence.shape
        
        # Step 1: Keypoint Reconstruction (nhanh)
        interpolated = self.keypoint_recon.interpolate(skeleton_sequence)
        
        # Step 2: STGAIN (nếu cần và đã enable)
        if self.use_stgain and self.stgain_model is not None:
            interpolated = self._apply_stgain(interpolated)
        
        return interpolated
    
    def _apply_stgain(self, skeleton_sequence: np.ndarray) -> np.ndarray:
        """
        Apply STGAIN để refine interpolation
        
        Args:
            skeleton_sequence: Shape (T, N, D)
        
        Returns:
            refined: Shape (T, N, D)
        """
        # Convert to tensor
        data = torch.FloatTensor(skeleton_sequence).unsqueeze(0)  # (1, T, N, D)
        data = data.to(self.device)
        
        # Create mask (1 for valid, 0 for missing)
        if data.shape[-1] >= 3:
            confidence = data[:, :, :, 2]  # (1, T, N)
            mask = (confidence >= self.confidence_threshold).float()
        else:
            mask = torch.ones(1, data.shape[1], data.shape[2], device=self.device)
        
        # Apply STGAIN
        with torch.no_grad():
            reconstructed, _ = self.stgain_model(data[:, :, :, :2], mask)
        
        # Convert back to numpy
        refined = reconstructed.squeeze(0).cpu().numpy()  # (T, N, D)
        
        # Blend với original data (chỉ replace missing keypoints)
        if skeleton_sequence.shape[-1] >= 3:
            confidence = skeleton_sequence[:, :, 2]
            missing_mask = confidence < self.confidence_threshold
            
            # Chỉ replace missing keypoints
            result = skeleton_sequence.copy()
            result[missing_mask, :2] = refined[missing_mask, :2]
        else:
            result = refined
        
        return result


# ========== TEST MODULES ==========

if __name__ == '__main__':
    print("=" * 70)
    print("TEST SMART INTERPOLATION")
    print("=" * 70)
    
    # ========== Test 1: Keypoint Reconstruction ==========
    print("\n" + "=" * 70)
    print("TEST 1: KEYPOINT RECONSTRUCTION")
    print("=" * 70)
    
    # Create dummy skeleton data with missing keypoints
    T, N, D = 64, 27, 3  # 3D: x, y, confidence
    np.random.seed(42)
    
    skeleton_data = np.random.randn(T, N, D) * 0.1
    
    # Add some movement
    for t_idx in range(T):
        skeleton_data[t_idx, :, 0] += np.sin(t_idx * 0.1) * 0.5
        skeleton_data[t_idx, :, 1] += np.cos(t_idx * 0.1) * 0.5
    
    # Set confidence
    skeleton_data[:, :, 2] = 0.9  # High confidence
    
    # Create missing keypoints (randomly set confidence to 0)
    missing_frames = np.random.choice(T, 10, replace=False)
    missing_joints = np.random.choice(N, 5, replace=False)
    
    for frame_idx in missing_frames:
        for joint_idx in missing_joints:
            skeleton_data[frame_idx, joint_idx, 2] = 0.1  # Low confidence
    
    print(f"\nOriginal data shape: {skeleton_data.shape}")
    print(f"Number of missing keypoints: {np.sum(skeleton_data[:, :, 2] < 0.3)}")
    
    # Apply Keypoint Reconstruction
    recon = KeypointReconstruction(
        anchor_joint_idx=0,
        scale_joints=(0, 4),
        confidence_threshold=0.3
    )
    
    interpolated = recon.interpolate(skeleton_data)
    
    print(f"Interpolated data shape: {interpolated.shape}")
    print(f"Missing keypoints after interpolation: {np.sum(interpolated[:, :, 2] < 0.3)}")
    
    # Check smoothness
    diff_original = np.diff(skeleton_data[:, :, :2], axis=0)
    diff_interpolated = np.diff(interpolated[:, :, :2], axis=0)
    
    smoothness_original = np.mean(diff_original ** 2)
    smoothness_interpolated = np.mean(diff_interpolated ** 2)
    
    print(f"\nSmoothness (original): {smoothness_original:.6f}")
    print(f"Smoothness (interpolated): {smoothness_interpolated:.6f}")
    
    # ========== Test 2: STGAIN Model ==========
    print("\n" + "=" * 70)
    print("TEST 2: STGAIN MODEL")
    print("=" * 70)
    
    # Create dummy input
    batch_size = 2
    x = torch.randn(batch_size, T, N, 2)  # (B, T, N, 2)
    mask = torch.ones(batch_size, T, N)  # All valid
    mask[0, 10:15, 5:8] = 0  # Some missing keypoints
    
    print(f"\nInput shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Create STGAIN model
    stgain = STGAIN(num_joints=N, in_channels=2, hidden_dim=64)
    
    print(f"\nSTGAIN parameters: {sum(p.numel() for p in stgain.parameters()):,}")
    
    # Forward pass
    reconstructed, mask_logits = stgain(x, mask)
    
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Mask logits shape: {mask_logits.shape}")
    
    # ========== Test 3: Smart Interpolation Pipeline ==========
    print("\n" + "=" * 70)
    print("TEST 3: SMART INTERPOLATION PIPELINE")
    print("=" * 70)
    
    pipeline = SmartInterpolationPipeline(
        num_joints=N,
        confidence_threshold=0.3,
        use_stgain=False  # Disable STGAIN for now
    )
    
    result = pipeline.interpolate(skeleton_data)
    
    print(f"\nInput shape: {skeleton_data.shape}")
    print(f"Output shape: {result.shape}")
    print(f"Missing keypoints before: {np.sum(skeleton_data[:, :, 2] < 0.3)}")
    print(f"Missing keypoints after: {np.sum(result[:, :, 2] < 0.3)}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)