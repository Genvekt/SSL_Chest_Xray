from models.pretraining.moco import ModifiedMocoV2
from data_loaders.data_module import ChestDataModule
from transforms.pretraining import Moco2TrainTransforms, Moco2ValTransforms

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

#===========================================================================
# ARGUMENTS
#===========================================================================

EXPERIMENT_NAME = "NL-chest14-gb7_flg-chest_xray_pneumonia-vinbigdata"
SEED = 1234


# DATA MODULE arguments
DS_LIST = ['chest14', 'gb7_flg', 'chest_xray_pneumonia', 'vinbigdata']
TRAIN_FRACTION = 1
BATCH_SIZE = 16
NUM_WORKERS = 2

# MOCO model arguments
MOCO_KWARGS = {
    'pretrained': True,
    'linear': False,
    'base_encoder': "resnet18", 
    'learning_rate': 0.0001,
    'num_negatives': 65536,
    'batch_size': BATCH_SIZE,
    'num_workers': NUM_WORKERS
}
#===========================================================================


def pretrain(ds_list, train_fraction, batch_size, num_workers, seed, model_kwargs):

    # Load data module
    dm = ChestDataModule(ds_list=ds_list, 
                         batch_size=batch_size, 
                         num_workers=num_workers, 
                         balanced=True, train_fraction=train_fraction,
                         seed=seed)

    dm.train_transforms = Moco2TrainTransforms(height=256)
    dm.val_transforms = Moco2TrainTransforms(height=256)

    # Load model
    model = ModifiedMocoV2(**model_kwargs)

    wandb_logger = WandbLogger(name='moco_pretrain_'+EXPERIMENT_NAME,project='thesis')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          dirpath='logs/pretraining/moco/', 
                                          filename=EXPERIMENT_NAME+'-{epoch:02d}-{val_loss:.4f}')

    trainer = pl.Trainer(gpus=1, deterministic=True,
                        logger=wandb_logger, callbacks=[checkpoint_callback])

    if torch.cuda.is_available():
        classifier = model.cuda()

    trainer.fit(model, dm)


if __name__ == "__main__":

    seed_everything(SEED)
    
    pretrain(ds_list=DS_LIST,
             train_fraction=TRAIN_FRACTION,
             batch_size=BATCH_SIZE,
             num_workers=NUM_WORKERS,
             seed=SEED,
             model_kwargs=MOCO_KWARGS)