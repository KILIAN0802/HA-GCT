import numpy as np
import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader
from utils.augmentation import SpatialAugmentation
from utils.preprocessing import SkeletonTransforms

class VSLDataset(Dataset):
    """Dataset cho 400VSL với Augmentation"""
    def __init__(self, data_path, label_path, transform=None, is_train=True):
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
                crop_min_ratio=0.6,
                crop_prob=0.5
            )
        else:
            self.augmentor = None
        
        print(f"Loaded {len(self.data)} samples ({'Train' if is_train else 'Test'})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # (C, T, V)
        label = self.labels[idx]
        
        # Apply Spatial Augmentation (chỉ khi train)
        if self.is_train and self.augmentor is not None:
            sample = self.augmentor(sample)
        
        # Apply other transforms (One Euro Filter, Normalization...)
        if self.transform:
            sample = self.transform(sample)
        
        if isinstance(sample, torch.Tensor):
            return sample.float(), torch.LongTensor([label]).squeeze()
        else:
            return torch.FloatTensor(sample), torch.LongTensor([label]).squeeze() #squeeze dùng để bỏ chiều dư thừa


def get_dataloaders(data_dir, batch_size=32, num_workers=4, transform=None):
    #Create train, val, test dataset  
    train_dataset = VSLDataset(f'{data_dir}/train_data_joint.npy', f'{data_dir}/train_label.pkl', transform=transform, is_train=True)
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
    def __init__(self, data_dir, transform=None, signer_ids=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        # List all npy files
        all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        
        # We need a stable mapping of word IDs to contiguous labels (0 to 198)
        # to prevent indexing errors.
        word_ids = sorted(list(set([int(f.split('_')[-1].replace('.npy', '')) for f in all_files])))
        self.word_to_label = {wid: idx for idx, wid in enumerate(word_ids)}
        
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
                crop_min_ratio=0.6,
                crop_prob=0.5
            )
        else:
            self.augmentor = None
            
        print(f"Loaded {len(self.files)} samples from {data_dir} ({'Train' if is_train else 'Val/Test'})")
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        filename, label = self.files[idx]
        file_path = os.path.join(self.data_dir, filename)
        
        # Load sample coordinates shape: (150, 27, 3)
        sample = np.load(file_path)
        
        # Apply Spatial Augmentation (chỉ khi train)
        if self.is_train and self.augmentor is not None:
            # SpatialAugmentation expects (C, T, V) but loaded sample is (T, V, C)
            sample = sample.transpose(2, 0, 1)
            sample = self.augmentor(sample)
            sample = sample.transpose(1, 2, 0)
            
        # Apply transform (One Euro Filter + Smart Interpolation + reshape to (2, max_frames, 27))
        if self.transform:
            sample = self.transform(sample)
            
        if isinstance(sample, torch.Tensor):
            return sample.float(), torch.tensor(label, dtype=torch.long)
        else:
            return torch.FloatTensor(sample), torch.tensor(label, dtype=torch.long)


def get_multivsl_loaders(data_dir, batch_size=32, num_workers=4, transform=None, split_method='random'):
    if split_method == 'signer':
        # Split by signer (Cross-signer evaluation)
        all_signers = ['02', '03', '04', '05', '06', '07-Phu', '08', '09', '10', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '26', '27', '28', '30', '31']
        train_signers = all_signers[:21]
        val_signers = all_signers[21:24]
        test_signers = all_signers[24:]
        
        train_dataset = MultiVSL200Dataset(data_dir, transform=transform, signer_ids=train_signers, is_train=True)
        val_dataset = MultiVSL200Dataset(data_dir, transform=transform, signer_ids=val_signers, is_train=False)
        test_dataset = MultiVSL200Dataset(data_dir, transform=transform, signer_ids=test_signers, is_train=False)
    else:
        # Random split (80% train, 10% val, 10% test)
        full_dataset = MultiVSL200Dataset(data_dir, transform=transform)
        total_len = len(full_dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        
        # Use random_split from torch
        from torch.utils.data import random_split
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )
        
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