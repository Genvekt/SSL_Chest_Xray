from pathlib import Path
from torchvision import transforms
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from data_loaders.chexpert_dataset import CheXpertDataset
from torch.utils.data import DataLoader, RandomSampler

class CheXpertDataModule(LightningDataModule):
    name = "CheXpert"
    def __init__(self, 
                 dataset_dir: str,
                 csv_path:str,
                 val_split: int = 0.2,
                 num_workers: int = 2,
                 batch_size: int = 16,
                 seed : int = 123456, 
                 target_class = None,
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
        self.DATASET = CheXpertDataset
        self.csv_path = csv_path
        self.dataset_dir = dataset_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.target_class = target_class
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
        if self.target_class is not None:
            return 2
        else:
            return 15
        
        
    def train_dataloader(self):
        """
        CheXpert train data
        """
        dataset = self.DATASET(self.dataset_dir,
                               csv_path=self.csv_path, 
                               transform=self.transform, 
                               part="train", 
                               target_class=self.target_class,
                               seed=self.seed)
        
        if self.train_fraction is not None:
            # Create sampler to get only fraction of test data
            sampler = RandomSampler(data_source=dataset, 
                                    num_samples=int(len(dataset) * self.train_fraction),
                                    replacement=True,
                                    generator=torch.Generator().manual_seed(self.seed))
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True
            )
        else:
            loader = DataLoader(
                dataset,
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
                               csv_path=self.csv_path, 
                               transform=self.transform, 
                               part="val", 
                               target_class=self.target_class,
                               seed=self.seed)
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader
    
    
    def test_dataloader(self):
        """
        CheXpert Temp test data (actually it is validation set)
        """

        dataset = self.DATASET(self.dataset_dir, 
                               csv_path=self.csv_path, 
                               transform=self.transform, 
                               part="val", 
                               target_class=self.target_class,
                               seed=self.seed)
       
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader