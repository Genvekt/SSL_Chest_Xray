from models.pretraining.moco import ModifiedMocoV2
from data_loaders.data_module import ChestDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.baseline import BaseLineClassifier
from transforms.finetuning import ChestTrainTransforms, ChestValTransforms

from pytorch_lightning import seed_everything
import pytorch_lightning as pl
import torch

#===========================================================================
# ARGUMENTS
#===========================================================================

EXPERIMENT_NAME = "NL_pneumo_0.1_Adam_3e-5"
SEED = 1234


# DATA MODULE arguments
DS_LIST = ['chest_xray_pneumonia']
TRAIN_FRACTION = 0.1
BATCH_SIZE = 16
NUM_WORKERS = 2

# BASELINE model arguments
BASELINE_KWARGS = {
    'num_classes': 2, 
    'linear': False,
    'learning_rate': 3e-5,
    'b1': 0.9,
    'b2': 0.999
}
#===========================================================================


def finetune(ds_list, train_fraction, batch_size, num_workers, seed, model_kwargs):

    # Load data module
    dm = ChestDataModule(ds_list=ds_list, 
                         batch_size=batch_size, 
                         num_workers=num_workers, 
                         balanced=True, train_fraction=train_fraction,
                         seed=seed)

    dm.train_transforms = ChestTrainTransforms(height=256)
    dm.val_transforms = ChestValTransforms(height=256)

    model = ModifiedMocoV2.load_from_checkpoint("logs/pretraining/moco/NL-chest14-gb7_flg-chest_xray_pneumonia-vinbigdata-epoch=14-val_loss=0.3542.ckpt")

    classifier = BaseLineClassifier(model.encoder_q, **model_kwargs)

    wandb_logger = WandbLogger(name='moco_finetune_'+EXPERIMENT_NAME,project='thesis')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          dirpath='logs/finetune/moco/', 
                                          filename=EXPERIMENT_NAME+'-{epoch:02d}-{val_loss:.4f}')


    trainer = pl.Trainer(gpus=1, deterministic=True,
                        logger=wandb_logger, callbacks=[checkpoint_callback])

    if torch.cuda.is_available():
        classifier = classifier.cuda()

    trainer.fit(classifier, dm)

if __name__ == "__main__":

    seed_everything(SEED)
    
    finetune(ds_list=DS_LIST,
             train_fraction=TRAIN_FRACTION,
             batch_size=BATCH_SIZE,
             num_workers=NUM_WORKERS,
             seed=SEED,
             model_kwargs=BASELINE_KWARGS)