import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

class CheXpertDataset(Dataset):
    
    def __init__(self, 
                 dataset_dir:str, 
                 csv_path:str,
                 transform=None, 
                 part="full", 
                 target_class=None, 
                 val_split=0.2,
                 seed = 12345):
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
        self.available_partitions = ["full", "train", "val", "train_val"]
        self.transform = transform
        dataset_dir = Path(dataset_dir)
        
        # Define pathes to all important files, read label data
        self.csv_data = pd.read_csv(csv_path)
        self.csv_data = self.csv_data[self.csv_data["Frontal/Lateral"]=="Frontal"]
        self.image_dir = dataset_dir.parent
        
        if part in ['train', 'val']:
            
            # Group by patient id
            self.csv_data["patient"] = self.csv_data["Path"].str.split('/').str[2]
            
            # Split to have patient images only in 1 partition
            train_data, val_data = train_test_split(
                np.array(list(self.csv_data.groupby(by=["patient"]).indices.items())), 
                test_size=val_split,
                random_state=seed, 
                shuffle=True)
            
            # Get needed partition
            if part == 'train':
                self.csv_data = self.csv_data.iloc[np.concatenate(train_data[:,1])]
            else:
                self.csv_data = self.csv_data.iloc[np.concatenate(val_data[:,1])]
            
 
        # Define label to idx mapping
        self.labels = [ "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
                       "Lung Opacity","Lung Lesion","Edema", "Consolidation",
                       "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
                       "Pleural Other", "Fracture", "Support Devices"]
        if target_class is not None:
            self.labels = [target_class]
            
        # Define label to idx mapping
        self.label_to_idx = {}
        for i,label in enumerate(self.labels):
            self.label_to_idx[label] = i
            
        # Define idx to label mapping
        self.idx_to_label = {}
        for label, idx in self.label_to_idx.items():
            self.idx_to_label[idx] = label

            
    def __len__(self):
        """
        Get the size of the dataset
        """
        return len(self.csv_data)
    
    
    def __getitem__(self, idx):
        """
        Get data item based on its index
        """
        # Convert torch tensors if given
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Read and preprocess image
        image_name = str(self.image_dir / self.csv_data["Path"].iloc[idx])
        image = Image.open(image_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get label and its one hot encoding
        if len(self.labels) == 1:
            one_hot_label = np.abs(np.nan_to_num(self.csv_data[self.labels].iloc[idx].to_list()))[0]
            sample = {'image': image, 
                      'one_hot_label':  torch.tensor(one_hot_label).long(), 
                      'label':int(one_hot_label),
                      'text_label':str(one_hot_label)}
            
        else:
            one_hot_label = np.abs(np.nan_to_num(csv_data[self.labels].iloc[idx].to_list()))
            sample = {'image': image.contiguous(), 
                      'one_hot_label':  torch.tensor(one_hot_label).long(), 
                      'label':int(one_hot_label),
                      'text_label':one_hot_label}
        
        return sample