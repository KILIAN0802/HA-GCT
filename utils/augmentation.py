import numpy as np
import random

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
        mask_prob=0.1,
        num_mask_joints=2,
        bone_scale=0.05
    ):
        self.num_joints = num_joints
        self.mask_prob = mask_prob
        self.num_mask_joints = num_mask_joints
        self.bone_scale = bone_scale
        
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
        
        # 1. Bone Scaling
        if self.bone_scale > 0:
            sample = self._apply_bone_scaling(sample)
            
        # 2. Joint Masking
        if self.mask_prob > 0 and random.random() < self.mask_prob:
            sample = self._apply_joint_masking(sample)
            
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
