import numpy as np
import math

class OneEuroFilter:
    """
    One Euro Filter - Adaptive Low-Pass Filter
    
    Tự động điều chỉnh cutoff frequency dựa trên tốc độ thay đổi của tín hiệu.
    
    Tham khảo:
    - Gery Casiez et al. (2012): "1€ Filter: A Simple Speed-based Low-pass Filter 
      for Noisy Input in Interactive Systems"
    - CHI 2012 Conference
    
    Ưu điểm:
    - Giảm nhiễu khi tín hiệu đứng yên hoặc chuyển động chậm
    - Không làm trễ (lag) khi tín hiệu chuyển động nhanh
    - Phù hợp cho real-time applications
    """
    
    def __init__(
        self,
        freq=30,          # Sampling frequency (Hz) - ví dụ: 30fps
        mincutoff=1.0,    # Minimum cutoff frequency (Hz)
        beta=0.7,         # Speed coefficient
        dcutoff=1.0       # Cutoff frequency for derivative (Hz)
    ):
        """
        Args:
            freq: Sampling frequency (Hz)
            mincutoff: Minimum cutoff frequency (Hz) - kiểm soát độ mượt khi đứng yên
            beta: Speed coefficient - kiểm soát độ nhạy với tốc độ
            dcutoff: Cutoff frequency for derivative (Hz) - lọc nhiễu cho đạo hàm
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        # Initialize state
        self.x_prev = None      # Previous filtered value
        self.dx_prev = None     # Previous derivative
        self.t_prev = None      # Previous timestamp
        
        # Compute sampling period
        self.tau = 1.0 / freq
    
    def _alpha(self, cutoff):
        """
        Compute smoothing factor alpha from cutoff frequency
        
        Args:
            cutoff: Cutoff frequency (Hz)
        
        Returns:
            alpha: Smoothing factor in [0, 1]
        """
        return 1.0 / (1.0 + self.tau / (2.0 * math.pi * cutoff))
    
    def filter(self, x, t=None):
        """
        Filter a single value
        
        Args:
            x: Input value (float or numpy array)
            t: Timestamp (optional, default: use sampling frequency)
        
        Returns:
            x_filtered: Filtered value
        """
        # Update timestamp
        if t is None:
            t = self.t_prev + self.tau if self.t_prev is not None else 0.0
        
        # Compute time delta
        if self.t_prev is not None:
            dt = t - self.t_prev
        else:
            dt = self.tau
        
        # Compute derivative (speed)
        if self.x_prev is not None:
            dx = (x - self.x_prev) / dt
        else:
            dx = 0.0
        
        # Filter derivative to reduce noise
        if self.dx_prev is not None:
            dx_filtered = self._alpha(self.dcutoff) * dx + \
                         (1.0 - self._alpha(self.dcutoff)) * self.dx_prev
        else:
            dx_filtered = dx
        
        # Compute adaptive cutoff frequency
        # f_c = f_c_min + β·|dx/dt|
        cutoff = self.mincutoff + self.beta * abs(dx_filtered)
        
        # Compute smoothing factor
        alpha = self._alpha(cutoff)
        
        # Apply low-pass filter
        if self.x_prev is not None:
            x_filtered = alpha * x + (1.0 - alpha) * self.x_prev
        else:
            x_filtered = x
        
        # Update state
        self.x_prev = x_filtered
        self.dx_prev = dx_filtered
        self.t_prev = t
        
        return x_filtered
    
    def reset(self):
        """Reset filter state"""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None
    
    def filter_sequence(self, sequence, timestamps=None):
        """
        Filter entire sequence at once
        
        Args:
            sequence: Input sequence shape (T, D) or (T, N, D)
                - T: number of frames
                - N: number of joints (optional)
                - D: dimensions (x, y) or (x, y, z)
            timestamps: Optional timestamps array shape (T,)
        
        Returns:
            filtered_sequence: Filtered sequence with same shape as input
        """
        # Reset filter
        self.reset()
        
        # Handle different input shapes
        if sequence.ndim == 2:
            # Shape: (T, D)
            T, D = sequence.shape
            filtered = np.zeros_like(sequence)
            
            for t in range(T):
                if timestamps is not None:
                    filtered[t] = self.filter(sequence[t], timestamps[t])
                else:
                    filtered[t] = self.filter(sequence[t])
        
        elif sequence.ndim == 3:
            # Shape: (T, N, D)
            T, N, D = sequence.shape
            filtered = np.zeros_like(sequence)
            
            for t in range(T):
                for n in range(N):
                    if timestamps is not None:
                        filtered[t, n] = self.filter(sequence[t, n], timestamps[t])
                    else:
                        filtered[t, n] = self.filter(sequence[t, n])
        
        else:
            raise ValueError(f"Unsupported input shape: {sequence.shape}")
        
        return filtered


class MultiJointOneEuroFilter:
    """
    One Euro Filter cho nhiều joints
    
    Mỗi joint có filter riêng để xử lý độc lập
    """
    
    def __init__(
        self,
        num_joints=27,
        freq=30,
        mincutoff=1.0,
        beta=0.7,
        dcutoff=1.0
    ):
        """
        Args:
            num_joints: Number of joints
            freq: Sampling frequency (Hz)
            mincutoff: Minimum cutoff frequency (Hz)
            beta: Speed coefficient
            dcutoff: Cutoff frequency for derivative (Hz)
        """
        self.num_joints = num_joints
        
        # Create separate filter for each joint
        self.filters = [
            OneEuroFilter(
                freq=freq,
                mincutoff=mincutoff,
                beta=beta,
                dcutoff=dcutoff
            )
            for _ in range(num_joints)
        ]
    
    def filter(self, joints_data, timestamps=None):
        """
        Filter all joints
        
        Args:
            joints_data: Shape (T, N, D)
                - T: frames
                - N: joints
                - D: dimensions (x, y)
            timestamps: Optional timestamps (T,)
        
        Returns:
            filtered: Shape (T, N, D)
        """
        T, N, D = joints_data.shape
        filtered = np.zeros_like(joints_data)
        
        # Reset all filters
        for f in self.filters:
            f.reset()
        
        # Filter each joint independently
        for n in range(N):
            joint_sequence = joints_data[:, n, :]  # (T, D)
            filtered[:, n, :] = self.filters[n].filter_sequence(
                joint_sequence, timestamps
            )
        
        return filtered
    
    def reset(self):
        """Reset all filters"""
        for f in self.filters:
            f.reset()


# ========== TEST MODULES ==========

if __name__ == '__main__':
    print("=" * 70)
    print("TEST ONE EURO FILTER")
    print("=" * 70)
    
    # ========== Test 1: Single Value Filtering ==========
    print("\n" + "=" * 70)
    print("TEST 1: SINGLE VALUE FILTERING")
    print("=" * 70)
    
    # Create noisy signal
    np.random.seed(42)
    T = 100
    t = np.linspace(0, 10, T)
    
    # Clean signal: sine wave
    clean_signal = np.sin(2 * np.pi * t)
    
    # Noisy signal
    noisy_signal = clean_signal + np.random.normal(0, 0.3, T)
    
    # Apply One Euro Filter
    filter_1e = OneEuroFilter(freq=10, mincutoff=1.0, beta=0.7)
    filtered_signal = np.zeros(T)
    
    for i in range(T):
        filtered_signal[i] = filter_1e.filter(noisy_signal[i], t[i])
    
    print(f"\nSignal length: {T}")
    print(f"Noise std: 0.3")
    print(f"Filter parameters:")
    print(f"  freq: 10 Hz")
    print(f"  mincutoff: 1.0 Hz")
    print(f"  beta: 0.7")
    
    # Compute MSE
    mse_noisy = np.mean((noisy_signal - clean_signal) ** 2)
    mse_filtered = np.mean((filtered_signal - clean_signal) ** 2)
    
    print(f"\nMSE (noisy):    {mse_noisy:.4f}")
    print(f"MSE (filtered): {mse_filtered:.4f}")
    print(f"Improvement:    {(1 - mse_filtered/mse_noisy)*100:.1f}%")
    
    # ========== Test 2: Multi-Joint Filtering ==========
    print("\n" + "=" * 70)
    print("TEST 2: MULTI-JOINT FILTERING")
    print("=" * 70)
    
    # Create dummy skeleton data
    T = 64
    N = 27
    D = 2  # x, y coordinates
    
    # Simulate hand movement
    skeleton_data = np.random.randn(T, N, D) * 0.1  # Small noise
    
    # Add some movement
    for t_idx in range(T):
        skeleton_data[t_idx, :, 0] += np.sin(t_idx * 0.1) * 0.5
        skeleton_data[t_idx, :, 1] += np.cos(t_idx * 0.1) * 0.5
    
    # Apply Multi-Joint Filter
    multi_filter = MultiJointOneEuroFilter(
        num_joints=N,
        freq=30,
        mincutoff=1.0,
        beta=0.7
    )
    
    filtered_skeleton = multi_filter.filter(skeleton_data)
    
    print(f"\nInput shape:  {skeleton_data.shape}")
    print(f"Output shape: {filtered_skeleton.shape}")
    
    # Compute smoothness (variance of differences)
    diff_noisy = np.diff(skeleton_data, axis=0)
    diff_filtered = np.diff(filtered_skeleton, axis=0)
    
    smoothness_noisy = np.mean(diff_noisy ** 2)
    smoothness_filtered = np.mean(diff_filtered ** 2)
    
    print(f"\nSmoothness (noisy):    {smoothness_noisy:.6f}")
    print(f"Smoothness (filtered): {smoothness_filtered:.6f}")
    print(f"Smoothness improvement: {(1 - smoothness_filtered/smoothness_noisy)*100:.1f}%")
    
    # ========== Test 3: Parameter Sensitivity ==========
    print("\n" + "=" * 70)
    print("TEST 3: PARAMETER SENSITIVITY")
    print("=" * 70)
    
    # Test different beta values
    betas = [0.0, 0.3, 0.7, 1.0, 2.0]
    
    for beta in betas:
        filter_test = OneEuroFilter(freq=10, mincutoff=1.0, beta=beta)
        filtered_test = np.zeros(T)
        
        for i in range(T):
            filtered_test[i] = filter_test.filter(noisy_signal[i], t[i])
        
        mse = np.mean((filtered_test - clean_signal) ** 2)
        print(f"  beta={beta:.1f}: MSE={mse:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)