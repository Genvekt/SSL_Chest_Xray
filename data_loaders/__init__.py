import torch
from pathlib import Path
import numpy as np
from torch.utils.data import random_split, Dataset
from data_loaders.chest_14_loader import Chest14Dataset
from data_loaders.rsna_loader import RSNADataset
from data_loaders.chest_pneumonia_loader import ChestPneumoniaDataset


DATASETS = {
    "RSNA": [Path("/datasets/rsna"), RSNADataset],
    "Chest14": [Path("/datasets/chest-14"), Chest14Dataset],
    "ChestPneumonia": [Path("/datasets/chest-xray-pneumonia"), ChestPneumoniaDataset],
}

VAL_SPLIT = 0.2
RANDOM_SEED = 1234


def get_data_loader(dataset_name:str, transform=None, part="full"):
    if dataset_name in DATASETS:
        dataset_dir, dataset_class = DATASETS[dataset_name]
        
        data_volume = dataset_class(dataset_dir, transform=transform, part=part)
        if part in ["train", "val"] and part not in data_volume.available_partitions:
            # Split "train_val" into "train" and "val"
            val_len = int(len(data_volume)*VAL_SPLIT)
            train_len = len(data_volume) - val_len 
            train_data, val_data = random_split(data_volume, 
                                                [train_len, val_len], 
                                                generator=torch.Generator().manual_seed(RANDOM_SEED))
            if part == "train":
                return train_data
            else:
                return val_data
        else:
            return data_volume
            
    else:
        print("Undefined dataset:", dataset_name)
        print("Available datasets:", list(DATASETS.keys()))
        return None
    
    
class RotationWrapper(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Randomly rotate image 
        image = sample["image"]
        angle = np.random.randint(4)
        image = image.rotate(90*angle)
        
        # Create rotation label
        one_hot_label = np.zeros(4)
        one_hot_label[angle] = 1
        
        if self.transform:
            image = self.transform(image)
            
        
        sample["image"] = image
        sample["one_hot_label"] = torch.tensor(one_hot_label).float()
        sample["label"] = torch.tensor(angle)
        
        return sample