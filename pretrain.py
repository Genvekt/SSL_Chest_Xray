from models import get_config
from itertools import product
from data_loaders.data_module import ChestDataModule
import hydra
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch


DATASETS = ['chest14']

MODEL_NAME = 'moco'

BASE_ENCODERS = ['resnet18']
LR = [0.00003]
LINEAR_OPTIONS = [True]



def pretrain(model_name, base_encoder, datasets, lr, linear=False):
    config = get_config(model_name)

    config.parameters['base_encoder'] = base_encoder
    config.parameters['lr'] = lr
    config.parameters['linear'] = linear

    model = hydra.utils.instantiate(config.model, **config.parameters)

    data_module = ChestDataModule(ds_list=datasets, batch_size=16, num_workers=2, balanced=True)

    train_transforms = hydra.utils.instantiate(config.transforms.train)
    val_transforms = hydra.utils.instantiate(config.transforms.val)

    data_module.train_transforms = train_transforms
    data_module.val_transforms = val_transforms

    linear_str = "linear" if linear else "nonlinear"
    ds_str = "-".join(datasets)
    lr_str = str(lr)

    save_dir = Path("logs/pretraining/"+model_name+"/"+base_encoder+"/")
    save_dir.mkdir(exist_ok=True, parents=True)
    save_file = "_".join([linear_str, ds_str, lr_str]) + '-{epoch:02d}-{val_loss:.4f}'

    wandb_logger = WandbLogger(name="_".join([model_name,base_encoder,linear_str,ds_str,lr_str]),project='thesis')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                        dirpath=str(save_dir),
                                        filename=save_file)

    trainer = pl.Trainer(gpus=1, deterministic=True,
                        logger=wandb_logger, callbacks=[checkpoint_callback])

    if torch.cuda.is_available():
        model = model.cuda()

    trainer.fit(model, data_module)


if __name__ == "__main__":

    for params in product(BASE_ENCODERS, LR, LINEAR_OPTIONS):

        base_encoder, lr, linear = params
        print("Processing next params: Encoder", base_encoder, ", lr", lr, ", linear", linear)
        pretrain(MODEL_NAME, base_encoder, DATASETS, lr, linear=linear)