from pl_bolts.models.self_supervised import MocoV2
import torch.nn.functional as F
from models import get_model, freeze_features, change_output
from pl_bolts.models.self_supervised.moco.moco2_module import precision_at_k
from pl_bolts.metrics import mean


class ModifiedMocoV2(MocoV2):
    def __init__(self, pretrained:bool=False, linear:bool=False, **kwargs):
        self.pretrained=pretrained
        self.linear=linear
        super().__init__(**kwargs)
        
    def init_encoders(self, base_encoder_name):
        # Load model
        encoder_q = get_model(base_encoder_name, pretrained=self.pretrained)
        encoder_k = get_model(base_encoder_name, pretrained=self.pretrained)
            
        # Substitute last layer with correct output dimention
        change_output(encoder_q, self.hparams.emb_dim)
        change_output(encoder_k, self.hparams.emb_dim)

        if self.linear:
            freeze_features(encoder_q)
            freeze_features(encoder_k)

        return encoder_q, encoder_k
    
    def training_step(self, batch, batch_idx):

        (img_1, img_2) = batch["image"]

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {
            'train_step_loss': loss,
            'train_step_acc': acc1,
            'train_step_acc5': acc5
        }
        self.log_dict(log)
        return {"loss":loss, 'train_loss': loss, 'train_acc': acc1, 'train_acc5': acc5,}
    
    def training_epoch_end(self, outputs):
        train_loss = mean(outputs, 'train_loss')
        train_acc = mean(outputs, 'train_acc')
        train_acc5 = mean(outputs, 'train_acc5')

        log = {
            'train_epoch_loss': train_loss,
            'train_epoch_acc': train_acc,
            'train_epoch_acc5': train_acc5,
            'train_lr': self.optimizers().param_groups[0]["lr"]
        }
        self.log_dict(log)

    def validation_step(self, batch, batch_idx):

        (img_1, img_2), labels = batch["image"], batch["target"]

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output, target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            'val_loss': loss,
            'val_acc': acc1,
            'val_acc5': acc5
        }
        return results
    
    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc = mean(outputs, 'val_acc')
        val_acc5 = mean(outputs, 'val_acc5')

        log = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_acc5': val_acc5
        }
        self.log_dict(log)