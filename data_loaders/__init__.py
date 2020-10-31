from pathlib import Path
from data_loaders.chest_14_loader import Chest14Dataset
from data_loaders.rsna_loader import RSNADataset
from data_loaders.chest_pneumonia_loader import ChestPneumoniaDataset


datasets = {
    "RSNA": [Path("/datasets/rsna"), RSNADataset],
    "Chest14": [Path("/datasets/chest-14"), Chest14Dataset],
    "ChestPneumonia": [Path("/datasets/chest-xray-pneumonia"), ChestPneumoniaDataset],
}


def get_data_loader(dataset_name:str, transform=None, part="full"):
    if dataset_name in datasets:
        dataset_dir, dataset_class = datasets[dataset_name]
        return dataset_class(dataset_dir, transform=transform, part=part)
    else:
        print("Undefined dataset:", dataset_name)
        print("Available datasets:", list(datasets.keys()))
        return None