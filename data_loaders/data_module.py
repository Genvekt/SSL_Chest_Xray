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
    def __init__(self, ds_list=None, batch_size=None, config=None, transform=None, balanced=False):
        self.transform = transform
        config = config if config else self._load_yaml_config()
        self.config = config.datasets
        self.ds_list = ds_list if ds_list else self.config.list
        self.batch_size = batch_size if batch_size else self.config.batch_size
        self.balanced = balanced

        # Check that names are valid
        self.ds_list = [ds_name for ds_name in self.ds_list if ds_name in self.config]
        print("Loaded datasets:", ",".join(self.ds_list))


    def create_dataloader(self, phase):
        datasets = []
        for ds_name in self.ds_list:
            ds_meta = self.config[ds_name]
            params = dict(ds_meta.dataset.parameters)

            csv_data = pd.read_csv(ds_meta.csv)
            csv_data = csv_data[csv_data["Phase"] == phase]

            params.update({
                    'csv_data': csv_data,
                    'transform': self.transform
                    })

            dataset = hydra.utils.instantiate(ds_meta.dataset.init, **params)
            datasets.append(dataset)

        datasets = ConcatDataset(datasets)

        dataloader_params = {}
        
        if self.balanced:
            dataloader_params.update(
                {"sampler": create_weighted_sampler(datasets, return_weights=False)})

        dataloader = DataLoader(
            datasets,
            batch_size=self.batch_size,
            **dataloader_params

        )
        return dataloader


    def train_dataloader(self):
        return self.create_dataloader(phase="train")


    def val_dataloader(self):
        return self.create_dataloader(phase="val")


    def test_dataloader(self):
        return self.create_dataloader(phase="test")


    def _load_yaml_config(self):
        """
        loads yaml config
        """
        config = pkg_resources.resource_stream(__name__, '../config/datasets.yaml')
        return DictConfig(yaml.safe_load(config))


