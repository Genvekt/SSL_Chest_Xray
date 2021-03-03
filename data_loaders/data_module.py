from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset
import pkg_resources
import yaml

from omegaconf import DictConfig

import pandas as pd
import hydra

from data_loaders.weighted_sampler import create_weighted_sampler

class ChestDataModule(LightningDataModule):
    def __init__(self, ds_list=None, 
                 batch_size=None, 
                 num_workers=None, 
                 config=None, 
                 balanced=False, 
                 train_fraction=None, 
                 seed=None, num_classes=2, return_dict=True):

        super(ChestDataModule, self).__init__()
        config = config if config else self._load_yaml_config()
        self.config = config.datasets
        self.ds_list = ds_list if ds_list else self.config.list
        self.batch_size = batch_size if batch_size else self.config.batch_size
        self.num_workers = num_workers if num_workers else self.config.num_workers
        self.balanced = balanced
        self.train_fraction = train_fraction
        self.seed = seed if seed else self.config.seed
        self.num_classes = num_classes
        self.return_dict = return_dict

        # Check that names are valid
        self.ds_list = [ds_name for ds_name in self.ds_list if ds_name in self.config]
        print("Loaded datasets:", ",".join(self.ds_list))

    def get_transform_by_phase(self, phase):
        if phase == "train":
            return self.train_transforms
        elif phase == "val":
            return self.val_transforms
        elif phase == "test":
            return self.test_transforms
        else:
            return None

    def create_dataloader(self, phase):
        datasets = []
        for ds_name in self.ds_list:
            ds_meta = self.config[ds_name]
            params = dict(ds_meta.dataset.parameters)

            csv_data = pd.read_csv(ds_meta.csv)
            csv_data = csv_data[csv_data["Phase"] == phase]
            print("Before sampling length: ", len(csv_data))

            # Sample train dataset if required.
            # Class balance is preserved as each class is sampled separately
            if phase == "train" and self.train_fraction is not None:
                if "Target" in csv_data:
                    csv_data = csv_data.groupby('Target', group_keys=False).apply(
                        lambda x: x.sample(int(len(x)*self.train_fraction), random_state=self.seed))
                else:
                    csv_data = csv_data.sample(int(len(csv_data)*self.train_fraction), random_state=self.seed)




            print("After sampling length: ", len(csv_data))
            transform = self.get_transform_by_phase(phase)

            params.update({
                    'csv_data': csv_data,
                    'transform': transform,
                    'return_dict': self.return_dict
                    })

            dataset = hydra.utils.instantiate(ds_meta.dataset.init, **params)
            datasets.append(dataset)

        datasets = ConcatDataset(datasets)

        dataloader_params = {}

        dataloader_params.update(
                {"sampler": None,
                 "shuffle": False,
                 "batch_size": self.batch_size,
                 "num_workers":self.num_workers,
                 "drop_last": True,
                 "pin_memory": True
                })
        
        if phase == "train":
            if self.balanced:
                print("Creating balanced dataloader")
                dataloader_params.update(
                    {"sampler": create_weighted_sampler(datasets, return_weights=False),
                    "shuffle": False})
            else:
                dataloader_params.update(
                    {"sampler": None,
                     "shuffle": True,
                    })
            
        dataloader = DataLoader(
            datasets,
            **dataloader_params
        )
        return dataloader


    def train_dataloader(self):
        return self.create_dataloader(phase="train")


    def val_dataloader(self):
        return self.create_dataloader(phase="val")


    def test_dataloader(self):
        return self.create_dataloader(phase="test")

    # def get_size(self, phase):
    #     total_len = 0
    #     for ds_name in self.ds_list:
    #         ds_meta = self.config[ds_name]
    #         csv_data = pd.read_csv(ds_meta.csv)
    #         csv_data = csv_data[csv_data["Phase"] == phase]

    #         total_len += len(csv_data)
            
    #     return total_len

    def _load_yaml_config(self):
        """
        loads yaml config
        """
        config = pkg_resources.resource_stream(__name__, '../config/datasets.yaml')
        return DictConfig(yaml.safe_load(config))


