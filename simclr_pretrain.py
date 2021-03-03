from pl_bolts.models.self_supervised.simclr.simclr_module import SimCLR
from data_loaders.data_module import ChestDataModule
import pytorch_lightning as pl
from transforms.pretraining import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pytorch_lightning.loggers import WandbLogger
import torch
from pytorch_lightning import seed_everything
seed_everything(12345)


dm = ChestDataModule(["chexpert_5"], batch_size=16, num_workers=2, balanced=False, return_dict=False)
dm.train_transforms = SimCLRTrainDataTransform(input_height=224,
                                               gaussian_blur=False,
                                               jitter_strength=0.5,
                                               normalize=True)

dm.val_transforms = SimCLREvalDataTransform(input_height=224,
                                            gaussian_blur=False,
                                            jitter_strength=0.5,
                                            normalize=True)

train_size = dm.get_size("train")
model = SimCLR(gpus=1, batch_size=16, num_samples=train_size, dataset='chexpert_5', learning_rate=0.5, warmup_epochs=1,
               arch='resnet18')




wandb_logger = WandbLogger(name='simclr-NL-chexpert5-full-resnet18-adamLARS-05',project='thesis')
checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                      dirpath='logs/pretraining/simclr/', 
                                      filename='NL-resnet18-chexpert5-full-adamLARS-05-{epoch:02d}-{val_loss:.4f}')

trainer = pl.Trainer(gpus=1, deterministic=True,
                     logger=wandb_logger, callbacks=[checkpoint_callback], max_epochs=100)

if torch.cuda.is_available():
    model = model.cuda()

trainer.fit(model, dm)