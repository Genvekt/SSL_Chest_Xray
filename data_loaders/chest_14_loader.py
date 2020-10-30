import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd

class Chest14Dataset(Dataset):
    
    def __init__(self, dataset_dir:Path, transform=None, part="full"):
        """
        Initialise chest-14 dataset
        URL: https://www.kaggle.com/nih-chest-xrays/data
        
        Args:
            dataset_dir (Path): path to dataset directory
            transform (callable, optional): Preprocessing transforms
            part (str): Type of partition
                       - "full" : Use the whole dataset
                       - "train_val": Use only train_val partition
                       - "test": Use only test partition
            
        """
        self.transform = transform
        
        # Define pathes to all important files, read label data
        self.csv_data = pd.read_csv(dataset_dir / "Data_Entry_2017.csv")
        self.image_dir = dataset_dir / "images"
 
        # Define label to idx mapping
        self.label_to_idx = {
            "No Finding": 0,
            "Atelectasis": 1,
            "Cardiomegaly": 2,
            "Effusion": 3,
            "Infiltration": 4,
            "Mass": 5,
            "Nodule": 6,
            "Pneumonia": 7,
            "Pneumothorax": 8,
            "Consolidation": 9,
            "Edema": 10,
            "Emphysema": 11,
            "Fibrosis": 12,
            "Pleural_Thickening": 13,
            "Hernia": 14
        }
        
        # Define idx to label mapping
        self.idx_to_label = {}
        for label, idx in self.label_to_idx.items():
            self.idx_to_label[idx] = label
        
        # Filter data based on defined partition type
        if part in ["train_val", "test"]:
            split_file = dataset_dir / (part + "_list.txt")
            image_names  = []
            with open(split_file, "r") as f:
                image_names = f.read().split("\n")
            self.csv_data = self.csv_data[self.csv_data['Image Index'].isin(image_names)]
            
            
    def label_to_one_hot(self, label_string):
        """
        Convert string label to one hot array
        """
        labels = label_string.split("|")
        one_hot_label = np.zeros(len(self.label_to_idx.keys()))
        for label in labels:
            idx = self.label_to_idx[label]
            one_hot_label[idx] = 1
        return one_hot_label
        
        
    def one_hot_to_label(self, one_hot_label):
        """
        Convert one hot array to string label
        """
        if len(one_hot_label) != len(self.label_to_idx.keys()):
            return "Undefined"
        else:
            labels = []
            for i in len(one_hot_label):
                if(one_hot_label[i]):
                    labels.append(self.idx_to_label[i])
            return "|".join(labels)

            
    def __len__(self):
        """
        Get the size of the dataset
        """
        return len(self.scv_data)
    
    
    def __getitem__(self, idx):
        """
        Get data item based on its index
        """
        # Convert torch tensors if given
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read and preprocess image
        img_name = self.image_dir / self.csv_data.iloc[idx, 0]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        
        # Get label and its one hot encoding
        label = self.csv_data.iloc[idx, 1]
        one_hot_label = self.label_to_one_hot(label)
        
        # Form output
        sample = {'image': image, 
                  'one_hot_label':  torch.tensor(one_hot_label).float(), 
                  'label':label}
        
        return sample