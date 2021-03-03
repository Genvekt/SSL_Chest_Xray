import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.metrics import mean
from models import get_model, freeze_features, change_output

    
class BaseLineClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, 
                 learning_rate=1e-4, 
                 b1=0.9,
                 b2=1e-4,
                 weight_decay=0,
                 linear=False, 
                 multi_class=False,
                 mixup=False,
                 ct_reg=False,
                 alpha = 0.3,
                 theta_low=0.35,
                 theta_high=0.75,
                 b_c = 0.2,
                 *args, **kwargs):
        super().__init__()
        self.multi_class = multi_class
        self.mixup = mixup
        self.ct_reg = ct_reg
        self.theta_low, self.theta_high = theta_low, theta_high
        self.b_c = b_c
        self.save_hyperparameters()
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()
        self.bce = torch.nn.BCELoss(reduction='none')
        self.beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
        change_output(self.model, num_classes)
        # Freeze everething but classifier
        if linear:
            freeze_features(self.model)
        self.accuracy = pl.metrics.Accuracy()
        
    def forward(self, x):
        return self.model(x)

    def compute_basic_loss(self, preds, target):
        if self.multi_class:
            return torch.mean(self.bce(self.sigmoid(preds), target))
        else:
            return F.cross_entropy(preds, target)

    def compute_mixup_loss(self, preds, target1, target2, lambdas, inv_lambdas):

        batch_size = lambdas.shape[0]

        lambdas = lambdas.reshape((batch_size,1))
        inv_lambdas = inv_lambdas.reshape((batch_size,1))

        if self.multi_class:
            preds = self.sigmoid(preds)

            self.compute_confidence_tempering(preds)

            loss1 = self.bce(preds, target1)
            loss2 = self.bce(preds, target2)
        else:
            loss1 = F.cross_entropy(preds, target1, reduction='none')
            loss2 = F.cross_entropy(preds, target2, reduction='none')

        total_loss = torch.sum(loss1*lambdas + loss2*inv_lambdas)

        return total_loss

    def compute_confidence_tempering(self, preds):

        if self.multi_class:
            probas = self.sigmoid(preds)
        else:
            probas = F.softmax(preds, dim=1)
        per_class = torch.mean(probas, dim=0)
        eps = 1e-8

        confidence_tempering = torch.sum(torch.log(self.theta_low/(per_class+eps) + per_class/self.theta_high))
        return confidence_tempering

    def apply_mixup(self, images, tagets):
        batch_size = images.shape[0]
        rolled_imgs = torch.roll(images, 1, 0)    
        rolled_targets = torch.roll(tagets, 1, 0)  

        lambdas = self.beta_distribution.sample((batch_size,1,1,1))
        lambdas = lambdas.type_as(images)
        inv_lambdas = 1 - lambdas

        resulted_imgs = images*lambdas + rolled_imgs*inv_lambdas

        return resulted_imgs, tagets, rolled_targets, lambdas, inv_lambdas
    
    def training_step(self, batch, batch_nb):
        x, target = batch["image"], batch["target"]
        if not self.mixup:
            preds = self(x)
            if isinstance(preds, list):
                preds = preds[0]

            loss = self.compute_basic_loss(preds, target)
            acc = self.accuracy(preds, target)

        else:
            mixed_imgs, targets1, targets2, lambdas, inv_lambdas = self.apply_mixup(x, target)
            
            preds = self(mixed_imgs)
            if isinstance(preds, list):
                preds = preds[0]

            loss = self.compute_mixup_loss(preds, targets1, targets2, lambdas, inv_lambdas)
            acc = torch.tensor(0.0).type_as(loss)

        if self.ct_reg:
            loss += self.b_c * self.compute_confidence_tempering(preds)

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
        x, target = batch["image"], batch["target"]

        preds = self(x)
        if isinstance(preds, list):
            preds = preds[0]

        loss = self.compute_basic_loss(preds, target)
        if self.multi_class:
            acc = torch.tensor(0.0).type_as(loss)
        else:
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


    def test_step(self, batch, batch_idx):
        x, target = batch["image"], batch["target"]

        preds = self(x)
        if isinstance(preds, list):
            preds = preds[0]

        loss = self.compute_basic_loss(preds, target)
        if self.multi_class:
            acc = torch.tensor(0.0).type_as(loss)
        else:
            acc = self.accuracy(preds, target)


        results = {
            'test_loss': loss,
            'test_acc': acc
        }
        return results


    def test_epoch_end(self, outputs):
        val_loss = mean(outputs, 'test_loss')
        val_acc = mean(outputs, 'test_acc')

        log = {
            'test_loss': val_loss,
            'test_acc': val_acc,
        }
        self.log_dict(log)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     self.hparams.learning_rate,
                                     betas=(self.hparams.b1, self.hparams.b2),
                                     weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3)
        return {"optimizer": optimizer,
                "lr_scheduler":scheduler,
                "monitor":'val_loss'}