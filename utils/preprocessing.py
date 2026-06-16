import numpy as np
import torch
from .one_euro_filter import MultiJointOneEuroFilter
from .smart_interpolation import SmartInterpolationPipeline

class SkeletonPreprocessor:
    """
    Enhanced Skeleton Data Preprocessing Pipeline
    
    Bao gồm:
    1. Smart Interpolation (Keypoint Reconstruction + STGAIN)
    2. One Euro Filter (lọc nhiễu)
    3. Normalization (chuẩn hóa)
    """
    
    def __init__(
        self,
        num_joints=27,
        freq=30,
        mincutoff=1.0,
        beta=0.7,
        dcutoff=1.0,
        confidence_threshold=0.3,
        use_stgain=False,
        reference_joint=0,
        scale_reference_joints=None,
        verbose=False
    ):
        """
        Args:
            num_joints: Number of joints
            freq: Sampling frequency (Hz)
            mincutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            dcutoff: Cutoff frequency for derivative (Hz)
            confidence_threshold: Threshold cho missing keypoints
            use_stgain: Có sử dụng STGAIN không
            reference_joint: Joint index cho root-relative normalization
            scale_reference_joints: Tuple của joint indices cho scale normalization
        """
        self.num_joints = num_joints
        self.reference_joint = reference_joint
        self.scale_reference_joints = scale_reference_joints
        self.verbose = verbose
        
        # Smart Interpolation
        self.interpolation = SmartInterpolationPipeline(
            num_joints=num_joints,
            confidence_threshold=confidence_threshold,
            use_stgain=use_stgain
        )
        
        # One Euro Filter
        self.filter = MultiJointOneEuroFilter(
            num_joints=num_joints,
            freq=freq,
            mincutoff=mincutoff,
            beta=beta,
            dcutoff=dcutoff
        )
    
    def preprocess(self, skeleton_data, timestamps=None, apply_filter=True):
        """
        Full preprocessing pipeline
        
        Args:
            skeleton_data: Shape (T, N, D)
            timestamps: Optional timestamps (T,)
            apply_filter: Có áp dụng One Euro Filter không
        
        Returns:
            preprocessed: Shape (T, N, D)
        """
        data = skeleton_data.copy()
        
        # Step 1: Smart Interpolation (missing keypoints)
        if self.verbose:
            print("Step 1: Smart Interpolation...")
        data = self.interpolation.interpolate(data)
        
        # Step 2: One Euro Filter (noise reduction)
        if apply_filter:
            if self.verbose:
                print("Step 2: One Euro Filter...")
            data = self.filter.filter(data, timestamps)
        
        # Step 3: Root-relative normalization
        if self.verbose:
            print("Step 3: Root-relative normalization...")
        data = self.normalize_root_relative(data)
        
        # Step 4: Scale normalization
        if self.scale_reference_joints:
            if self.verbose:
                print("Step 4: Scale normalization...")
            data = self.normalize_scale(data)
        
        return data
    
    def normalize_root_relative(self, skeleton_data):
        """Root-relative normalization"""
        root_coords = skeleton_data[:, self.reference_joint, :]
        normalized = skeleton_data - root_coords[:, np.newaxis, :]
        return normalized
    
    def normalize_scale(self, skeleton_data):
        """Scale normalization"""
        if self.scale_reference_joints is None:
            return skeleton_data
        
        joint1 = skeleton_data[:, self.scale_reference_joints[0], :]
        joint2 = skeleton_data[:, self.scale_reference_joints[1], :]
        
        distances = np.sqrt(np.sum((joint1 - joint2) ** 2, axis=-1))
        distances = np.maximum(distances, 1e-6)
        
        normalized = skeleton_data / distances[:, np.newaxis, np.newaxis]
        
        return normalized


class SkeletonTransforms:
    """
    Skeleton Transforms Wrapper for PyTorch/torchvision pipelines.
    
    Transforms raw skeleton coordinates into HA-GCT network format.
    """
    def __init__(
        self,
        num_joints=27,
        max_frames=64,
        freq=30,
        mincutoff=1.0,
        beta=0.7,
        dcutoff=1.0,
        confidence_threshold=0.3,
        use_stgain=False,
        stgain_model_path=None,
        verbose=False
    ):
        self.num_joints = num_joints
        self.max_frames = max_frames
        
        self.preprocessor = SkeletonPreprocessor(
            num_joints=num_joints,
            freq=freq,
            mincutoff=mincutoff,
            beta=beta,
            dcutoff=dcutoff,
            confidence_threshold=confidence_threshold,
            use_stgain=use_stgain,
            verbose=verbose
        )
        
        if use_stgain and stgain_model_path is not None:
            self.preprocessor.interpolation.load_stgain_model(stgain_model_path)
            
    def __call__(self, coords, confidence=None):
        """
        Args:
            coords: numpy array of shape:
                    - (C, T, V): From VSLDataset
                    - (T, N, 2) or (T, N, 3): From raw MediaPipe input
            confidence: numpy array of shape (T, N) (optional)
            
        Returns:
            transformed_tensor: torch.Tensor of shape (2, max_frames, 27)
        """
        # If coords is passed as a tuple/list (coords, confidence)
        if isinstance(coords, (tuple, list)):
            if len(coords) == 2:
                coords, confidence = coords
        
        # If coords has shape (C, T, V), transpose to (T, V, C)
        if coords.ndim == 3 and coords.shape[0] in [2, 3]:
            coords = coords.transpose(1, 2, 0)  # Shape: (T, V, C)
            
        # Extract or construct (T, N, 3) for preprocessor
        if confidence is not None:
            T, N, _ = coords.shape
            confidence_expanded = np.expand_dims(confidence, axis=-1)
            skeleton_data = np.concatenate([coords[:, :, :2], confidence_expanded], axis=-1)
        else:
            if coords.shape[-1] == 2:
                T, N, _ = coords.shape
                confidence_expanded = np.ones((T, N, 1), dtype=coords.dtype)
                skeleton_data = np.concatenate([coords, confidence_expanded], axis=-1)
            else:
                skeleton_data = coords.copy()
                
        # Run preprocessing (Interpolation, One Euro Filter, Normalization)
        preprocessed = self.preprocessor.preprocess(skeleton_data, apply_filter=True)
        
        # Extract X, Y coordinates
        coords_out = preprocessed[:, :, :2]  # Shape: (T, N, 2)
        
        # Adjust temporal dimension to max_frames (pad or crop)
        T_curr, N_curr, C_curr = coords_out.shape
        if T_curr != self.max_frames:
            if T_curr > self.max_frames:
                coords_out = coords_out[:self.max_frames]
            else:
                pad_width = ((0, self.max_frames - T_curr), (0, 0), (0, 0))
                coords_out = np.pad(coords_out, pad_width, mode='constant')
                
        # Permute to (C, T, N) shape = (2, max_frames, N)
        coords_tensor = torch.from_numpy(coords_out).float()
        coords_tensor = coords_tensor.permute(2, 0, 1).contiguous()
        
        return coords_tensor


# ========== TEST MODULES ==========
if __name__ == '__main__':
    print("=" * 70)
    print("TEST SKELETON PREPROCESSING PIPELINE & TRANSFORMS")
    print("=" * 70)
    
    # Generate mock MediaPipe data: (T=50, N=27, C=3) -> X, Y, confidence
    T_len = 50
    num_joints = 27
    np.random.seed(42)
    
    # Simulate a coordinate signal with noise
    t = np.linspace(0, 10, T_len)
    coords = np.zeros((T_len, num_joints, 2))
    for v in range(num_joints):
        coords[:, v, 0] = np.sin(t) + np.random.normal(0, 0.1, T_len)
        coords[:, v, 1] = np.cos(t) + np.random.normal(0, 0.1, T_len)
        
    confidence = np.ones((T_len, num_joints))
    # Make some values missing (confidence < 0.3)
    missing_idx = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    for idx in missing_idx:
        confidence[idx, :] = 0.1
        
    print(f"Input coordinates shape: {coords.shape} (T, N, 2)")
    print(f"Input confidence shape: {confidence.shape} (T, N)")
    print(f"Number of missing frames: {len(missing_idx)}")
    
    # Initialize transform
    transform = SkeletonTransforms(
        num_joints=num_joints,
        max_frames=64,
        verbose=True  # Turn on logging for validation
    )
    
    # Run transform
    out_tensor = transform(coords, confidence)
    print(f"Output tensor shape: {out_tensor.shape} (C, T, N)")
    assert out_tensor.shape == (2, 64, num_joints), "Incorrect transform shape!"
    
    # Check that there are no NaNs
    assert not torch.isnan(out_tensor).any(), "Found NaNs in output!"
    print("\nALL PREPROCESSING PIPELINE TESTS PASSED!")
    print("=" * 70)