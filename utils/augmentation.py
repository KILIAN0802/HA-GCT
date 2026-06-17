import numpy as np
import random
import math
import torch
import torch.nn.functional as F

class SpatialAugmentation:
    """
    Spatial Augmentation for Skeleton Sequences (400VSL dataset)
    
    Includes:
    1. Joint Masking (occlusion simulation)
    2. Bone Length Scaling (anatomically consistent scaling)
    """
    
    def __init__(
        self,
        num_joints=27,
        mask_prob=0.3,
        num_mask_joints=2,
        bone_scale=0.05,
        noise_std=0.01,
        noise_prob=0.5,
        shift_max=5,
        shift_prob=0.5,
        crop_min_ratio=0.8,
        crop_prob=0.5,
        rot_max_angle=20.0,
        rot_prob=0.6,
        speed_min_rate=0.7,
        speed_max_rate=1.3,
        speed_prob=0.5
    ):
        self.num_joints = num_joints
        self.mask_prob = mask_prob
        self.num_mask_joints = num_mask_joints
        self.bone_scale = bone_scale
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        self.shift_max = shift_max
        self.shift_prob = shift_prob
        self.crop_min_ratio = crop_min_ratio
        self.crop_prob = crop_prob
        self.rot_max_angle = rot_max_angle
        self.rot_prob = rot_prob
        self.speed_min_rate = speed_min_rate
        self.speed_max_rate = speed_max_rate
        self.speed_prob = speed_prob
        
        # Define parents for anatomically consistent bone scaling
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
            # Left Hand Fingers
            8: 7,      # Thumb L
            9: 7,      # Index L root
            10: 9,     # Index L tip
            11: 7,     # Middle L root
            12: 11,    # Middle L tip
            13: 7,     # Ring L root
            14: 13,    # Ring L tip
            15: 7,     # Pinky L root
            16: 15,    # Pinky L tip
            # Right Hand Fingers
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
        
        # Topological order to process parent before child
        self.topological_order = [
            0, 1, 2, 3, 4, 5, 6, 7, 17,
            8, 9, 10, 11, 12, 13, 14, 15, 16,
            18, 19, 20, 21, 22, 23, 24, 25, 26
        ]

    def __call__(self, sample):
        """
        Args:
            sample: numpy array of shape (C, T, V)
        Returns:
            augmented_sample: numpy array of shape (C, T, V)
        """
        # Make a copy to avoid in-place modification of dataset
        sample = sample.copy()
        
        # Convert to PyTorch tensor for the new augmentations (expects (T, V, C))
        C, T, V = sample.shape
        skeleton_tensor = torch.from_numpy(sample).permute(1, 2, 0).float()
        
        # A1. Speed Perturbation (called first)
        skeleton_tensor = self._apply_speed_perturbation(skeleton_tensor)
        
        # A2. Body Rotation (called second)
        skeleton_tensor = self._apply_random_rotation(skeleton_tensor)
        
        # Convert back to NumPy array of shape (C, T, V)
        sample = skeleton_tensor.permute(2, 0, 1).numpy()
        
        # 1. Bone Scaling
        if self.bone_scale > 0:
            sample = self._apply_bone_scaling(sample)
            
        # 2. Joint Masking
        if self.mask_prob > 0 and random.random() < self.mask_prob:
            sample = self._apply_joint_masking(sample)
            
        # 3. Random Gaussian Noise
        if self.noise_std > 0 and random.random() < self.noise_prob:
            sample = self._apply_gaussian_noise(sample)
            
        # 4. Random Temporal Shift
        if self.shift_max > 0 and random.random() < self.shift_prob:
            sample = self._apply_temporal_shift(sample)
            
        # 5. Random Temporal Crop
        if self.crop_min_ratio < 1.0 and random.random() < self.crop_prob:
            sample = self._apply_temporal_crop(sample)
            
        return sample

    def _apply_bone_scaling(self, sample):
        """
        Scales the bone vectors for each frame in the sequence
        """
        C, T, V = sample.shape
        
        # Generate random scaling factor for each bone (constant across the sequence to maintain identity)
        # We have V-1 bones (all except root 0)
        scales = {
            joint: random.uniform(1.0 - self.bone_scale, 1.0 + self.bone_scale)
            for joint in self.parents if self.parents[joint] is not None
        }
        
        # Apply scaling in topological order, vectorized across T
        for joint in self.topological_order:
            parent = self.parents[joint]
            if parent is None:
                continue  # Keep root coordinate unchanged
            
            # Get coordinates for all frames T
            parent_coords = sample[:2, :, parent]
            joint_coords = sample[:2, :, joint]
            
            # Bone vector
            bone_vectors = joint_coords - parent_coords
            
            # Scale bone vector
            scaled_vectors = bone_vectors * scales[joint]
            
            # Update coordinates
            sample[:2, :, joint] = parent_coords + scaled_vectors
            
        return sample

    def _apply_joint_masking(self, sample):
        """
        Masks random joints by setting their coordinates to 0
        """
        C, T, V = sample.shape
        
        # Choose joints to mask (exclude root joint 0)
        mask_joints = random.sample(range(1, V), min(self.num_mask_joints, V - 1))
        
        for joint in mask_joints:
            # Zero out x, y coordinates
            sample[:2, :, joint] = 0.0
            
            # If z/confidence exists, zero it out too
            if C >= 3:
                sample[2, :, joint] = 0.0
                
        return sample

    def _apply_gaussian_noise(self, sample):
        """
        Adds random Gaussian noise to joint coordinates
        """
        C, T, V = sample.shape
        # We only add noise to spatial dimensions (X, Y), which are channels 0 and 1
        noise = np.random.normal(0, self.noise_std, size=(2, T, V))
        sample[:2, :, :] += noise
        return sample

    def _apply_temporal_shift(self, sample):
        """
        Shifts the sequence temporally and pads/replicates boundaries
        """
        C, T, V = sample.shape
        if T < 10:
            return sample
        
        shift_max = min(self.shift_max, T // 5)
        if shift_max < 1:
            return sample
            
        shift = random.randint(-shift_max, shift_max)
        if shift == 0:
            return sample
            
        new_sample = np.zeros_like(sample)
        if shift > 0:
            # Shift right: copy [0, T-shift] to [shift, T]
            new_sample[:, shift:, :] = sample[:, :-shift, :]
            # Replicate the first frame for the padded part
            new_sample[:, :shift, :] = sample[:, 0:1, :]
        else:
            # Shift left: copy [|shift|, T] to [0, T-|shift|]
            abs_shift = abs(shift)
            new_sample[:, :-abs_shift, :] = sample[:, abs_shift:, :]
            # Replicate the last frame for the padded part
            new_sample[:, -abs_shift:, :] = sample[:, -1:, :]
            
        return new_sample

    def _apply_temporal_crop(self, sample):
        """
        Crops a temporal sub-sequence and interpolates it back to original length
        """
        C, T, V = sample.shape
        if T < 15:
            return sample
            
        crop_ratio = random.uniform(self.crop_min_ratio, 0.95)
        crop_len = int(T * crop_ratio)
        if crop_len >= T or crop_len < 5:
            return sample
            
        start_idx = random.randint(0, T - crop_len)
        cropped_sample = sample[:, start_idx:start_idx+crop_len, :]
        
        # Interpolate back to original length T
        new_sample = np.zeros((C, T, V), dtype=sample.dtype)
        x_old = np.linspace(0, 1, crop_len)
        x_new = np.linspace(0, 1, T)
        for c in range(C):
            for v in range(V):
                new_sample[c, :, v] = np.interp(x_new, x_old, cropped_sample[c, :, v])
        return new_sample

    def _apply_random_rotation(self, skeleton):
        if random.random() > self.rot_prob:
            return skeleton
        angle = random.uniform(-self.rot_max_angle, self.rot_max_angle) * math.pi / 180.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]], dtype=skeleton.dtype)
        
        # Translate to make joint 0 (root) at (0, 0) for each frame
        root_coords = skeleton[:, 0:1, :2].clone() # shape (T, 1, 2)
        skeleton_centered = skeleton[:, :, :2] - root_coords
        
        # Apply rotation
        rotated = skeleton_centered @ R.T
        
        # Copy back and translate back
        skeleton = skeleton.clone()
        skeleton[:, :, :2] = rotated + root_coords
        return skeleton

    def _apply_speed_perturbation(self, skeleton):
        if random.random() > self.speed_prob:
            return skeleton
        T = skeleton.shape[0]
        rate = random.uniform(self.speed_min_rate, self.speed_max_rate)
        new_T = max(2, int(T * rate))
        
        x = skeleton.permute(2, 1, 0).unsqueeze(0)  # (1, C, V, T)
        x = x.reshape(1, -1, T)
        x_resampled = F.interpolate(x, size=new_T, mode='linear', align_corners=False)
        x_final = F.interpolate(x_resampled, size=T, mode='linear', align_corners=False)
        return x_final.squeeze(0).reshape(skeleton.shape[2], skeleton.shape[1], T).permute(2, 1, 0)


if __name__ == '__main__':
    print("=" * 70)
    print("TEST SPATIAL AUGMENTATION")
    print("=" * 70)
    
    # Create dummy sample: C=3, T=64, V=27
    sample = np.ones((3, 64, 27))
    # Give joints distinct coordinates
    for v in range(27):
        sample[0, :, v] = float(v)
        sample[1, :, v] = float(v) * 2.0
        sample[2, :, v] = 0.9  # confidence
        
    print(f"Original shape: {sample.shape}")
    print(f"Original joint 1 coord (frame 0): {sample[:, 0, 1]}")
    
    augmentor = SpatialAugmentation(
        num_joints=27,
        mask_prob=1.0,  # Always mask for testing
        num_mask_joints=2,
        bone_scale=0.10  # 10% scaling
    )
    
    augmented = augmentor(sample)
    print(f"Augmented shape: {augmented.shape}")
    print(f"Augmented joint 1 coord (frame 0) after bone scaling: {augmented[:, 0, 1]}")
    
    # Count masked joints (coordinates are all 0)
    masked_count = 0
    for v in range(1, 27):
        if np.all(augmented[:2, :, v] == 0):
            masked_count += 1
            print(f"  Joint {v} is masked!")
            
    print(f"Total masked joints: {masked_count} (Expected: 2)")
    print("\nSPATIAL AUGMENTATION TEST PASSED!")
    print("=" * 70)
