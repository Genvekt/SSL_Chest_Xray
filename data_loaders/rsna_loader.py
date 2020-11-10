import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import pydicom

class RSNADataset(Dataset):
    
    def __init__(self, dataset_dir:Path, transform=None, part="full"):
        """
        Initialise RSNA dataset
        URL: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
        
        Args:
            dataset_dir (Path): path to dataset directory
            transform (callable, optional): Preprocessing transforms
            part (str): Type of partition
                       - "full" : Use the whole dataset, test has no labels
                       - "train_val": Use only train_val partition
                       - "test": Use only test partition, has no labels
        """
        self.available_partitions = ["full", "train_val", "test"]
        self.transform = transform
        if part in ["train", "val"]:
            part = "train_val"
        self.part = part
        
        # Train Partition
        self.train_image_dir = dataset_dir / "stage_2_train_images"
        self.train_csv = []
        if part in ["full", "train_val"]:
            csv_file = dataset_dir / "stage_2_train_labels.csv"
            self.train_csv = pd.read_csv(csv_file).drop_duplicates(subset=['patientId'])
        
        #Test partition
        self.test_image_dir = dataset_dir / "stage_2_test_images"
        self.test_csv = []
        if part in ["full", "test"]:
            csv_file = dataset_dir / "stage_2_sample_submission.csv"
            self.test_csv = pd.read_csv(csv_file)
        
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
        return len(self.train_csv) + len(self.test_csv)
    
    
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
        if self.part in ["full", "train_val"]:
            if idx < len(self.train_csv):
                image_path = self.train_image_dir / (self.train_csv.iloc[idx, 0]+'.dcm')
                label = self.idx_to_label[self.train_csv.iloc[idx, 5]]
            else:
                idx = idx - len(self.train_csv)
                image_path = self.test_image_dir / (self.test_csv.iloc[idx, 0]+'.dcm')
        else:
            image_path = self.test_image_dir / (self.test_csv.iloc[idx, 0]+'.dcm')
        image = pydicom.dcmread(image_path)
        image = Image.fromarray(image.pixel_array.astype(np.uint8))
        if self.transform:
            image = self.transform(image)
            
        label = self.label_to_idx[label]
        # Convert label to one hot, will be all zeros if label is None (for test data)   
        #one_hot_label = self.label_to_one_hot(label)
        
#         sample = {'image': image, 
#                   'one_hot_label':  torch.tensor(one_hot_label).float(), 
#                   'label':label}
        return image, label
        
        
        
            
            
        
        