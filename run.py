import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import data_loaders
import transforms
from pathlib import Path
from utils.visualisation import showInRow
from models import get_model

from data_loaders.data_module import ChestDataModule

@hydra.main(config_path="config", config_name="datasets.yaml")
def pipeline(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load target dataset for pretraining
    
    dm = ChestDataModule(cfg)
    dm.train_dataloader()


if __name__ == "__main__":
    pipeline()