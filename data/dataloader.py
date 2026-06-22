import numpy as np
import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader
from utils.augmentation import SpatialAugmentation
from utils.preprocessing import SkeletonTransforms

class VSLDataset(Dataset):
    """Dataset cho 400VSL với Augmentation"""
    def __init__(self, data_path, label_path, transform=None, is_train=True, crop_min_ratio=0.6):
        self.data = np.load(data_path)
        with open(label_path, 'rb') as f:
            self.labels = pickle.load(f)
        self.transform = transform
        self.is_train = is_train
        
        # Handle different formats
        if self.data.ndim == 4:
            if self.data.shape[1] in [2, 3]:
                pass
            elif self.data.shape[2] in [2, 3]:
                self.data = self.data.transpose(0, 3, 1, 2)
        
        # Initialize Augmentation (chỉ áp dụng khi train)
        if is_train:
            self.augmentor = SpatialAugmentation(
                num_joints=27,
                mask_prob=0.5,
                num_mask_joints=4,
                bone_scale=0.15,
                noise_std=0.04,
                noise_prob=0.7,
                shift_max=15,
                shift_prob=0.7,
                crop_min_ratio=crop_min_ratio,
                crop_prob=0.5
            )
        else:
            self.augmentor = None
        
        # Pre-apply transform (deterministic preprocessing) to the entire dataset to optimize speed
        if self.transform is not None:
            print(f"Pre-applying transform to the entire VSLDataset ({'Train' if is_train else 'Val/Test'})...")
            preprocessed_data = []
            valid_lengths = []
            for i in range(len(self.data)):
                res = self.transform(self.data[i])
                if isinstance(res, tuple):
                    coords_t, val_len = res
                else:
                    coords_t = res
                    val_len = coords_t.shape[1]
                preprocessed_data.append(coords_t)
                valid_lengths.append(val_len)
            # Stack into a tensor
            self.data = torch.stack(preprocessed_data)
            self.valid_lengths = torch.tensor(valid_lengths, dtype=torch.long)
            self.transform = None  # Clear transform so it isn't applied twice
        
        print(f"Loaded {len(self.data)} samples ({'Train' if is_train else 'Test'})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # (C, T, V) or preprocessed tensor
        label = self.labels[idx]
        
        # Apply Spatial Augmentation (chỉ khi train)
        if self.is_train and self.augmentor is not None:
            if isinstance(sample, torch.Tensor):
                sample_np = sample.numpy()
            else:
                sample_np = sample
            sample_np = self.augmentor(sample_np)
            sample = torch.from_numpy(sample_np).float()
        
        # Apply other transforms (One Euro Filter, Normalization...)
        valid_length = sample.shape[1] if isinstance(sample, torch.Tensor) else sample.shape[1] # fallback
        if self.transform:
            res = self.transform(sample)
            if isinstance(res, tuple):
                sample, valid_length = res
            else:
                sample = res
                valid_length = sample.shape[1]
        elif hasattr(self, 'valid_lengths'):
            valid_length = int(self.valid_lengths[idx].item())
            
        # Create boolean mask
        max_frames = sample.shape[1]
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:valid_length] = True
        
        if isinstance(sample, torch.Tensor):
            sample_tensor = sample.float()
        else:
            sample_tensor = torch.FloatTensor(sample)
            
        return sample_tensor, mask, torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_dir, batch_size=32, num_workers=4, transform=None, crop_min_ratio=0.6):
    #Create train, val, test dataset  
    train_dataset = VSLDataset(f'{data_dir}/train_data_joint.npy', f'{data_dir}/train_label.pkl', transform=transform, is_train=True, crop_min_ratio=crop_min_ratio)
    val_dataset = VSLDataset(f'{data_dir}/val_data_joint.npy', f'{data_dir}/val_label.pkl', transform=transform, is_train=False)
    test_dataset = VSLDataset(f'{data_dir}/test_data_joint.npy', f'{data_dir}/test_label.pkl', transform=transform, is_train=False)
    
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
        
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    return train_loader, val_loader, test_loader


class MultiVSL200Dataset(Dataset):
    """Dataset for MultiVSL200 loading individual npy files"""
    def __init__(self, data_dir, transform=None, signer_ids=None, is_train=True, crop_min_ratio=0.6, files=None):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # List all npy files
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        
        # We need a stable mapping of word IDs to contiguous labels (0 to 198)
        # to prevent indexing errors.
        word_ids = sorted(list(set([int(f.split('_')[-1].replace('.npy', '')) for f in all_files])))
        self.word_to_label = {wid: idx for idx, wid in enumerate(word_ids)}
        
        if files is not None:
            self.files = files
        else:
            self.files = []
            for f in all_files:
                signer_id = f.split('_')[0]
                word_id = int(f.split('_')[-1].replace('.npy', ''))
                label = self.word_to_label[word_id]
                
                # Filter by signer IDs if provided (useful for Cross-Signer validation/splits)
                if signer_ids is not None:
                    if signer_id in signer_ids:
                        self.files.append((f, label))
                else:
                    self.files.append((f, label))
                
        # Initialize Augmentation (chỉ áp dụng khi train)
        if is_train:
            self.augmentor = SpatialAugmentation(
                num_joints=27,
                mask_prob=0.5,
                num_mask_joints=4,
                bone_scale=0.15,
                noise_std=0.04,
                noise_prob=0.7,
                shift_max=15,
                shift_prob=0.7,
                crop_min_ratio=crop_min_ratio,
                crop_prob=0.5
            )
        else:
            self.augmentor = None
            
        print(f"Preloading and preprocessing {len(self.files)} samples from {data_dir} ({'Train' if is_train else 'Val/Test'})...")
        self.samples = []
        self.valid_lengths = []
        self.labels = []
        for f, label in self.files:
            file_path = os.path.join(self.data_dir, f)
            sample = np.load(file_path)
            sample = sample.transpose(2, 0, 1)
            # Pre-apply transform (interpolation, One Euro Filter, normalizations)
            valid_length = sample.shape[1] # fallback
            if self.transform is not None:
                res = self.transform(sample)
                if isinstance(res, tuple):
                    sample, valid_length = res
                else:
                    sample = res
                    valid_length = sample.shape[1]
                
            if isinstance(sample, torch.Tensor):
                sample = sample.numpy()
                
            self.samples.append(sample)
            self.valid_lengths.append(valid_length)
            self.labels.append(label)
            
        # Clear transform since we pre-applied it
        self.transform = None
        print(f"Loaded and preprocessed {len(self.samples)} samples.")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]  # Preprocessed tensor of shape (2, max_frames, 27)
        label = self.labels[idx]
        
        # Apply Spatial Augmentation (chỉ khi train)
        if self.is_train and self.augmentor is not None:
            # SpatialAugmentation expects numpy array of shape (C, T, V)
            if isinstance(sample, torch.Tensor):
                sample_np = sample.numpy()
            else:
                sample_np = sample
            sample_np = self.augmentor(sample_np)
            sample = torch.from_numpy(sample_np).float()
            
        # Get valid length
        if hasattr(self, 'valid_lengths') and len(self.valid_lengths) > idx:
            valid_length = self.valid_lengths[idx]
        else:
            valid_length = sample.shape[1]
            
        # Create boolean mask
        max_frames = sample.shape[1]
        mask = torch.zeros(max_frames, dtype=torch.bool)
        mask[:valid_length] = True
        
        if isinstance(sample, torch.Tensor):
            sample_tensor = sample.float()
        else:
            sample_tensor = torch.FloatTensor(sample)
            
        return sample_tensor, mask, torch.tensor(label, dtype=torch.long)


def get_multivsl_loaders(data_dir, batch_size=32, num_workers=4, transform=None, split_method='random', crop_min_ratio=0.6):
    if split_method == 'signer':
        # Split by signer (Cross-signer evaluation)
        all_signers = ['02', '03', '04', '05', '06', '07-Phu', '08', '09', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '26', '27', '28', '30', '31']
        train_signers = all_signers[:21]
        val_signers = all_signers[21:24]
        test_signers = all_signers[24:]
        
        train_dataset = MultiVSL200Dataset(data_dir, transform=transform, signer_ids=train_signers, is_train=True, crop_min_ratio=crop_min_ratio)
        val_dataset = MultiVSL200Dataset(data_dir, transform=transform, signer_ids=val_signers, is_train=False)
        test_dataset = MultiVSL200Dataset(data_dir, transform=transform, signer_ids=test_signers, is_train=False)
    else:
        # Random split (80% train, 10% val, 10% test)
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        word_ids = sorted(list(set([int(f.split('_')[-1].replace('.npy', '')) for f in all_files])))
        word_to_label = {wid: idx for idx, wid in enumerate(word_ids)}
        
        files_with_labels = []
        for f in all_files:
            word_id = int(f.split('_')[-1].replace('.npy', ''))
            label = word_to_label[word_id]
            files_with_labels.append((f, label))
            
        import random
        rng = random.Random(42)
        rng.shuffle(files_with_labels)
        
        total_len = len(files_with_labels)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        
        train_files = files_with_labels[:train_len]
        val_files = files_with_labels[train_len:train_len+val_len]
        test_files = files_with_labels[train_len+val_len:]
        
        train_dataset = MultiVSL200Dataset(data_dir, transform=transform, is_train=True, crop_min_ratio=crop_min_ratio, files=train_files)
        val_dataset = MultiVSL200Dataset(data_dir, transform=transform, is_train=False, files=val_files)
        test_dataset = MultiVSL200Dataset(data_dir, transform=transform, is_train=False, files=test_files)
        
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True
    }
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2
        
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data_dir = 'data'
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=32, num_workers=4)

    print("Train loader:", len(train_loader))
    print("Val loader:", len(val_loader))
    print("Test loader:", len(test_loader))