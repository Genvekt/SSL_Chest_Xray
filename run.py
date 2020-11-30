import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import data_loaders
import transforms

@hydra.main(config_path="configs", config_name="config.yaml")
def pipeline(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Load target dataset for pretraining

    target_transform = instantiate(cfg.target_transform)
    dataset = instantiate(cfg.target_dataset)
    dataset.transform = target_transform
   
    

if __name__ == "__main__":
    pipeline()