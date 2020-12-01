import torch
from pathlib import Path
import numpy as np
from torch.utils.data import random_split, Dataset
from data_loaders.chest_14_data_module import Chest14DataModule
from data_loaders.chexpert_data_module import CheXpertDataModule

DATASETS = {
    #"RSNA": [Path("/datasets/rsna"), RSNADataset],
    "Chest14": [Path("/datasets/chest-14"), Chest14DataModule],
    "CheXpert": [Path("/new_data/CheXpert/CheXpert-v1.0"), CheXpertDataModule]
#     "ChestPneumonia": [Path("/datasets/chest-xray-pneumonia"), ChestPneumoniaDataset],
}


def get_data_module(dataset_name:str, **kwargs):
    if dataset_name in DATASETS:
        dataset_dir, datamodule_class = DATASETS[dataset_name]
        
        return datamodule_class(dataset_dir, **kwargs)
            
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