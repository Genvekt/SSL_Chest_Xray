from pl_bolts.models.self_supervised.cpc import CPCV2
import torch
import math

class CPCV2Modified(CPCV2):

    def __init__(self, patched=True,**kwargs):
        super().__init__(**kwargs)
        self.patched = patched

    def shared_step(self, batch):

        img_1 = batch["image"]

        # generate features
        # Latent features
        Z = self(img_1)

        # infoNCE loss
        nce_loss = self.contrastive_task(Z)
        return nce_loss


    def init_encoder(self):
        encoder = super().init_encoder()
        self.hparams.encoder = encoder
        return encoder

    def forward(self, img_1):
        # put all patches on the batch dim for simultaneous processing
        if self.patched:
            b, p, c, w, h = img_1.size()
            img_1 = img_1.view(-1, c, w, h)

        # Z are the latent vars
        Z = self.encoder(img_1)

        if isinstance(Z, list):
            Z = Z[0]

        if self.patched:
            # (?) -> (b, -1, nb_feats, nb_feats)
            Z = self.__recover_z_shape(Z, b)

        return Z

    def __recover_z_shape(self, Z, b):
        # recover shape
        Z = Z.squeeze(-1)
        nb_feats = int(math.sqrt(Z.size(0) // b))
        Z = Z.view(b, -1, Z.size(1))
        Z = Z.permute(0, 2, 1).contiguous()
        Z = Z.view(b, -1, nb_feats, nb_feats)
        return Z
        


    