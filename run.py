import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import data_loaders
import transforms

@hydra.main(config_path="./configs", config_name="config.yaml")
def pipeline(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load target dataset for pretraining
    if
    target_transform = instantiate(cfg.target_transform)
    dataset = instantiate(cfg.target_dataset)
    print(dataset.__class__)
    print(dataset.transform.__class__)
    dataset.transform = target_transform
    print(dataset.transform.__class__)


if __name__ == "__main__":
    pipeline()