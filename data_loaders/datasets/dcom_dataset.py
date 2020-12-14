import pandas as pd 
from pathlib import Path

import torch
from torch.utils.data import Dataset

import numpy as np

import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut, apply_modality_lut

class DcomDataset(Dataset):
    
    def __init__(self, csv_data, transform=None, invert=False):
        """
        Basic dataset
        
        Args:
            csv_data: pandas.Dataframe of path to csv file.
            transform (callable, optional): Preprocessing transforms
            invert (bool): If true, black will be white and vise versa.
        """
       # self.dataset_dir = Path(dataset_dir) if not isinstance(dataset_dir, Path) else dataset_dir
        self.transform = transform
        self.csv_data = pd.read_csv(csv_data) if not isinstance(csv_data, pd.DataFrame) else csv_data 
        self.invert = invert
    
    def __len__(self):
        """
        Get the size of the dataset
        """
        return len(self.csv_data)
    
    def get_all_targets(self):
        return self.csv_data['Target'].values
    
    def __getitem__(self, idx):
        """
        Extract item 
        """
        label, img_path = self._get_label_and_path(idx)
        img = self.get_image(img_path)
        sample = {}
        sample["image"] = img
        sample["path"] = img_path
        sample["target"] = label

        if self.transform:
            sample["image"]  = self.transform(sample["image"])

        sample["image"] = self._img_to_tensor(sample['image'])
        return sample

    def _get_label_and_path(self, idx):
        """
        Extract the label and path to image by the idx
        """
        row = self.csv_data.iloc[idx]
        img_path = row['Path']
        label = self._process_target(row['Target'])

        return label, img_path

    def get_image(self, img_path):
        img = self._read_image(img_path)
        img = self._process_image(img)
        return img

    def _read_image(self, img_path):
        """
        Read image from specified path
        """
        dcm = pydicom.read_file(img_path)
        img = apply_modality_lut(apply_voi_lut(dcm.pixel_array, dcm), dcm)

        return img

    def _process_image(self, img):
        img = img.astype(np.float32)
        img -= img.min()
        img /= img.max()
        img = (img*255)
        img = img.astype(np.uint8)

        # inverting if required
        if hasattr(self, 'invert') and self.invert:
            img = np.invert(img)

        if (len(img.shape) == 3) and (img.shape[-1] == 3):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        img = img.astype(np.uint8)

        return img

    def _process_target(self, target):
        target = (
            target.float()
            if isinstance(target, torch.Tensor)
            else torch.tensor(target).float()
        )
        target = torch.unsqueeze(target, -1)

        return target

    def _img_to_tensor(self, img):
         return img.float() if isinstance(img, torch.Tensor) else torch.tensor(img).float()
