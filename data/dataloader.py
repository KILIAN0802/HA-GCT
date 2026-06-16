import numpy as np
import pickle
import torch
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
                mask_prob=0.1,        # 10% chance to mask joints
                num_mask_joints=2,    # Mask 2 joints
                bone_scale=0.05       # ±5% bone length
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data_dir = 'data'
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=32, num_workers=4)

    print("Train loader:", len(train_loader))
    print("Val loader:", len(val_loader))
    print("Test loader:", len(test_loader))