from data_loaders.data_module import ChestDataModule
from utils.visualisation import showInRow
from models import get_model

from transforms.pretraining import Moco2TrainTransforms, Moco2ValTransforms
from transforms.finetuning import ChestTrainTransforms, ChestValTransforms

from models.baseline import BaseLineClassifier

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
seed_everything(12345)


dm = ChestDataModule(["chexpert_5_01"], batch_size=16, num_workers=2, balanced=False)
dm.train_transforms = ChestTrainTransforms(height=224)
dm.val_transforms = ChestValTransforms(height=224)

classifier = BaseLineClassifier(get_model("resnet18", pretrained=True), 
                                num_classes=5, 
                                linear=False,
                                learning_rate=1e-4,
                                b1=0.9,
                                b2=0.999,
                                weight_decay=1e-4,
                                multi_class=True,
                                mixup=False,
                                ct_reg=False)

#classifier = BaseLineClassifier.load_from_checkpoint("logs/baseline/chexpert5/NL-01-mixup-ct_reg-Adam-1e_4-epoch=02-val_loss=0.5106.ckpt")

wandb_logger = WandbLogger(name='baseline-NL-chexpert5-01-Adam-1e_4',project='thesis')
checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                      dirpath='logs/baseline/chexpert5/', 
                                      filename='NL-01-Adam-1e_4-{epoch:02d}-{val_loss:.4f}')

trainer = pl.Trainer(gpus=1, deterministic=True,
                     logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=15)

if torch.cuda.is_available():
    classifier = classifier.cuda()

trainer.fit(classifier, dm)