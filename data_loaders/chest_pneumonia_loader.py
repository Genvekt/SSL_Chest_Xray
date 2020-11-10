import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class ChestPneumoniaDataset(Dataset):
    def __init__(self, dataset_dir:Path, transform=None, part="full"):
        """
        Initialise Chest Preumonia dataset
        URL: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
        
        Args:
            dataset_dir (Path): path to dataset directory
            transform (callable, optional): Preprocessing transforms
            part (str): Type of partition
                       - "full" : Use the whole dataset
                       - "train_val": Use both train and val partition
                       - "train": Use only train partition
                       - "val" : Use only val [artition
                       - "test": Use only test partition
        """
        self.available_partitions = ["full", "train_val", "train", "val", "test"]
        self.transform = transform
        
        self.pneumo_imgs = []
        self.norm_imgs = []
        
        if part in ["full", "train_val", "train"]:
            pneumo_dir = dataset_dir / "train/PNEUMONIA"
            norm_dir = dataset_dir / "train/NORMAL"
            self.pneumo_imgs += list(pneumo_dir.glob("*.jpeg"))
            self.norm_imgs += list(norm_dir.glob("*.jpeg"))
            
        if part in ["full", "train_val", "val"]:
            pneumo_dir = dataset_dir / "val/PNEUMONIA"
            norm_dir = dataset_dir / "val/NORMAL"
            self.pneumo_imgs += list(pneumo_dir.glob("*.jpeg"))
            self.norm_imgs += list(norm_dir.glob("*.jpeg"))
            
        if part in ["full", "test"]:
            pneumo_dir = dataset_dir / "test/PNEUMONIA"
            norm_dir = dataset_dir / "test/NORMAL"
            self.pneumo_imgs += list(pneumo_dir.glob("*.jpeg"))
            self.norm_imgs += list(norm_dir.glob("*.jpeg"))
        
        self.label_to_idx = {
            "No Pneumonia": 0,
            "Pneumonia": 1
        }
        self.idx_to_label = {
            0: "No Pneumonia",
            1: "Pneumonia"
        }
        
        
    def label_to_one_hot(self, label):
        """
        Convert string label to one hot array
        """
        one_hot_label = np.zeros(len(self.label_to_idx))
        if label is not None:
            idx = self.label_to_idx[label]
            one_hot_label[idx] = 1
        return one_hot_label
        
        
    def one_hot_to_label(self, one_hot_label):
        """
        Convert one hot array to string label
        """
        if len(one_hot_label) != len(self.label_to_idx) or not 1 in one_hot_label:
            return "Undefined"
        else:
            return self.idx_to_label[np.argmax(one_hot_label)]
    
        
        
    def __len__(self):
        """
        Get the size of the dataset
        """
        return len(self.pneumo_imgs) + len(self.norm_imgs)
    
    
    def __getitem__(self, idx):
        """
        Get data item based on its index
        """
        # Convert torch tensors if given
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image_path = None
        label = None
        
        # Get image and label
        if idx < len(self.pneumo_imgs):
            label = "Pneumonia"
            image_name = self.pneumo_imgs[idx]
        else:
            label = "No Pneumonia"
            image_name = self.norm_imgs[idx-len(self.pneumo_imgs)]
        
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        one_hot_label = self.label_to_one_hot(label)
        
        # Form output
        sample = {'image': image, 
                  'one_hot_label':  torch.tensor(one_hot_label).float(), 
                  'label':label}
        
        return sample