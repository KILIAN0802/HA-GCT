import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

class VSLDataset(Dataset):

    def __init__(self, data_path, label_path):
        #Load data
        self.data = np.load(data_path)

        #Load label
        with open(label_path, 'rb') as f:
            self.labels = pickle.load(f)

        # Print shape info
        print(f"Loaded {len(self.data)} video samples")
        print(f"Video shape: {self.data.shape[1:]}")
        print(f"Number of action classes: {len(np.unique(self.labels))}")

        #Handle different data formats
        if self.data.ndim == 4:
            #Format: (N, C, T, V) -> (N, T, V, C)
            if self.data.shape[1] in [2, 3]: #(N, C, T, V)
                pass
            elif self.data.shape[3] in [2, 3]: #(N, T, V, C)
                self.data = self.data.transpose(0, 3, 1, 2)
            else:
                raise ValueError(f"Unknown data format: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        return torch.FloatTensor(sample), torch.LongTensor([label]).squeeze() #squeeze dùng để bỏ chiều dư thừa


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    #Create train, val, test dataset  
    train_dataset = VSLDataset(f'{data_dir}/train_data_joint.npy', f'{data_dir}/train_label.pkl')
    val_dataset = VSLDataset(f'{data_dir}/val_data_joint.npy', f'{data_dir}/val_label.pkl')
    test_dataset = VSLDataset(f'{data_dir}/test_data_joint.npy', f'{data_dir}/test_label.pkl')
    
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