from models.pretraining.moco import ModifiedMocoV2
from data_loaders.data_module import ChestDataModule
from transforms.pretraining import Moco2TrainTransforms

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from pytorch_lightning import seed_everything
import pytorch_lightning as pl

#===========================================================================
# ARGUMENTS
#===========================================================================

EXPERIMENT_NAME = "NL-chexpert-chest14-chest_xray_pneumonia-gb7_flg-tbx11k-vinbigdata-full-part2from02epoch"
SEED = 1234


# DATA MODULE arguments
DS_LIST = ['chexpert', 'chest14', 'chest_xray_pneumonia', 'gb7_flg', 'tbx11k', 'vinbigdata']
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
    'num_workers': NUM_WORKERS,
    # Augmentations
    'height':224,
    'rotation':15,
    'translate':0.05,
    'scale':0.05,
    'square_pad': True,
    'normalisation':'imagenet',
    # Dataset info
    'seed':SEED,
    'datasets': DS_LIST,
    'train_fraction': TRAIN_FRACTION
}
#===========================================================================


def pretrain(ds_list, train_fraction, batch_size, num_workers, seed, model_kwargs):

    # Load data module
    dm = ChestDataModule(ds_list=ds_list, 
                         batch_size=batch_size, 
                         num_workers=num_workers, 
                         balanced=False, train_fraction=train_fraction,
                         seed=seed)

    dm.train_transforms = Moco2TrainTransforms(height=224)
    dm.val_transforms = Moco2TrainTransforms(height=224)

    # Load model
    #model = ModifiedMocoV2(**model_kwargs)
    model = ModifiedMocoV2.load_from_checkpoint("logs/pretraining/moco/NL-chexpert-chest14-chest_xray_pneumonia-gb7_flg-tbx11k-vinbigdata-full-epoch=02-val_loss=0.4410.ckpt")

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