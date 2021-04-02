import cv2
import torch
from data_loaders.datasets.cv2_dataset import CV2Dataset

class ChexpertDataset(CV2Dataset):
    
    def __init__(self, targets=None, **kwargs):
        """
        Initiate CheXpert Dataset
        """
        super(ChexpertDataset, self).__init__(**kwargs)

        if targets is None:
            self.targets = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        else:
            self.targets = list(targets)


    def _get_label_and_path(self, idx):
        """
        Extract the label and path to image by the idx
        """
        row = self.csv_data.iloc[idx]
        img_path = row['Path']
        label = self._process_target(row[self.targets])
        return label, img_path

    def _process_target(self, target):
        target = (
            target.float()
            if isinstance(target, torch.Tensor)
            else torch.tensor(target).float()
        )
        return target

    def get_all_targets(self):
        return self.csv_data[self.targets].values