import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.metrics import mean
from models import get_model, freeze_features, change_output


class BaseLineClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-4,momentum=0.9,weight_decay=1e-4,
                 linear=False,*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        change_output(self.model, num_classes)
        # Freeze everething but classifier
        if linear:
            freeze_features(self.model)
        self.accuracy = pl.metrics.Accuracy()
        
    def forward(self, x):
        return self.model(x) 
    
    def training_step(self, batch, batch_nb):
        x, target = batch["image"], batch["label"]
        preds = self(x)
        loss = F.cross_entropy(preds, target)
        acc = self.accuracy(preds, target)
        log = {
            'train_step_loss': loss,
            'train_step_acc': acc
        }
        self.log_dict(log)
        
        return {"loss":loss, 'train_loss': loss, 'train_acc': acc}
    
    
    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, 'train_loss')
        train_acc = mean(outputs, 'train_acc')

        log = {
            'train_epoch_loss': train_loss,
            'train_epoch_acc': train_acc,
            'train_lr': self.optimizers().param_groups[0]["lr"]
        }
        self.log_dict(log)
   
    def validation_step(self, batch, batch_idx):
        x, target = batch["image"], batch["label"]

        preds = self(x)
        loss = F.cross_entropy(preds, target)
        acc = self.accuracy(preds, target)


        results = {
            'val_loss': loss,
            'val_acc': acc
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc = mean(outputs, 'val_acc')

        log = {
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        self.log_dict(log)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.1, threshold=0.001)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
    
class BaseLineClassifierAdam(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate=1e-4, b1=0.9,b2=1e-4,
                 linear=False,*args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        change_output(self.model, num_classes)
        # Freeze everething but classifier
        if linear:
            freeze_features(self.model)
        self.accuracy = pl.metrics.Accuracy()
        
    def forward(self, x):
        return self.model(x) 
    
    def training_step(self, batch, batch_nb):
        x, target = batch["image"], batch["label"]
        preds = self(x)
        loss = F.cross_entropy(preds, target)
        acc = self.accuracy(preds, target)
        log = {
            'train_step_loss': loss,
            'train_step_acc': acc
        }
        self.log_dict(log)
        
        return {"loss":loss, 'train_loss': loss, 'train_acc': acc}
    
    
    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, 'train_loss')
        train_acc = mean(outputs, 'train_acc')

        log = {
            'train_epoch_loss': train_loss,
            'train_epoch_acc': train_acc,
            'train_lr': self.optimizers().param_groups[0]["lr"]
        }
        self.log_dict(log)
   
    def validation_step(self, batch, batch_idx):
        x, target = batch["image"], batch["label"]

        preds = self(x)
        loss = F.cross_entropy(preds, target)
        acc = self.accuracy(preds, target)


        results = {
            'val_loss': loss,
            'val_acc': acc
        }
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc = mean(outputs, 'val_acc')

        log = {
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        self.log_dict(log)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate,betas=(self.hparams.b1, self.hparams.b2))
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.1, threshold=0.001)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return {"optimizer": optimizer}