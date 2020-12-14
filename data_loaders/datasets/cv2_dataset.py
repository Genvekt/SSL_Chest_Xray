import cv2
from data_loaders.datasets.dcom_dataset import DcomDataset

class CV2Dataset(DcomDataset):
    
    def __init__(self, **kwargs):
        """
        Initiate CheXpert Dataset
        """
        super(CV2Dataset, self).__init__(**kwargs)
            
    def _read_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img