from models.pretraining.cpcv2 import CPCV2Modified
from data_loaders.data_module import ChestDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning import seed_everything

from transforms.pretraining import CPCTrainTransforms, CPCValTransforms
import pytorch_lightning as pl
import torch



#===========================================================================
# ARGUMENTS
#===========================================================================

EXPERIMENT_NAME = "vinbigdata_full_patch-64_overlap_32"
SEED = 1234


# DATA MODULE arguments
DS_LIST = ['vinbigdata']
TRAIN_FRACTION = 1
BATCH_SIZE = 16
NUM_WORKERS = 2
PATCH_SIZE = 64
OVERLAP = 32
HEIGHT = 256

# MOCO model arguments
CPC_KWARGS = {
    'pretrained': False,
    'encoder_name': "resnet18",
    'learning_rate': 0.0001,
    'num_classes':2,
    'patch_size': PATCH_SIZE,
    'patch_overlap':OVERLAP,
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

    dm.train_transforms = CPCTrainTransforms(patch_size=PATCH_SIZE, overlap=OVERLAP, height=HEIGHT)
    dm.val_transforms = CPCValTransforms(patch_size=PATCH_SIZE, overlap=OVERLAP, height=HEIGHT)

    # Load model
    model_kwargs['num_classes'] = dm.num_classes
    model = CPCV2Modified(**model_kwargs)

    wandb_logger = WandbLogger(name='cpc_pretrain_'+EXPERIMENT_NAME,project='thesis')
    checkpoint_callback = ModelCheckpoint(monitor='val_nce', 
                                          dirpath='logs/pretraining/cpc/', 
                                          filename=EXPERIMENT_NAME+'-{epoch:02d}-{val_nce:.4f}')

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
             model_kwargs=CPC_KWARGS)