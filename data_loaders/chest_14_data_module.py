from pathlib import Path

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader, random_split, RandomSampler


from data_loaders.chest_14_dataset import Chest14Dataset

class Chest14DataModule(LightningDataModule):
    name = "chest14"
    def __init__(self, 
                 dataset_dir: Path,
                 val_split: int = 0.2,
                 num_workers: int = 2,
                 batch_size: int = 16,
                 seed : int = 123456, 
                 binary: bool = True,
                 transform = None,
                 train_fraction = None,
                 *args,
                 **kwargs,):
        """
        Args:
            val_split
            num_workers
            batch_size
            seed
        """
        super().__init__(*args, **kwargs)
        self.DATASET = Chest14Dataset
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.binary = binary
        self.val_split = val_split
        self.transform = self.default_transforms() if transform is None else transform
        self.train_fraction = train_fraction
    
    
    def default_transforms(self):
        """
        Default transforms, only convert images to tensors
        """
        return transforms.Compose([transforms.ToTensor()])
    
    
    @property
    def num_classes(self):
        """
        Return number of available classes
        """
        if self.binary:
            return 2
        else:
            return 15
        
        
    def train_dataloader(self):
        """
        Chest14 train data
        """

        dataset = self.DATASET(self.dataset_dir, 
                               transform=self.transform, 
                               part="train", 
                               binary=self.binary)
        
        if "train" not in dataset.available_partitions:
            # Split "train_val" into "train" and "val"
            val_len = int(len(dataset)*self.val_split)
            train_len = len(dataset) - val_len 
            train_data, _ = random_split(dataset, 
                                         [train_len, val_len], 
                                         generator=torch.Generator().manual_seed(self.seed))
        if self.train_fraction is not None:
            # Create sampler to get only fraction of test data
            sampler = RandomSampler(data_source=train_data, 
                                    num_samples=int(len(train_data) * self.train_fraction),
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(self.seed))
            loader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            loader = DataLoader(
                train_data,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        return loader
    
    
    def val_dataloader(self):
        """
        Chest14 val data
        """

        dataset = self.DATASET(self.dataset_dir, 
                               transform=self.transform, 
                               part="val", 
                               binary=self.binary)
        
        if "val" not in dataset.available_partitions:
            # Split "train_val" into "train" and "val"
            val_len = int(len(dataset)*self.val_split)
            train_len = len(dataset) - val_len 
            _ , val_data = random_split(dataset, 
                                        [train_len, val_len], 
                                        generator=torch.Generator().manual_seed(self.seed))
        loader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader
    
    
    def test_dataloader(self):
        """
        Chest14 test data
        """

        dataset = self.DATASET(self.dataset_dir, 
                               transform=self.transform, 
                               part="test", 
                               binary=self.binary)
       
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader
        
    