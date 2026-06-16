import numpy as np
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
        scale_reference_joints=None
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
        print("Step 1: Smart Interpolation...")
        data = self.interpolation.interpolate(data)
        
        # Step 2: One Euro Filter (noise reduction)
        if apply_filter:
            print("Step 2: One Euro Filter...")
            data = self.filter.filter(data, timestamps)
        
        # Step 3: Root-relative normalization
        print("Step 3: Root-relative normalization...")
        data = self.normalize_root_relative(data)
        
        # Step 4: Scale normalization
        if self.scale_reference_joints:
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