from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')


class ChestTrainTransforms:
    """
    Moco 2 modified  augmentation:
    Classic: https://arxiv.org/pdf/2003.04297.pdf
    Chest:   https://arxiv.org/pdf/2010.05352v1.pdf
    """


    def __init__(self, height: int = 128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((height,height), ),
            transforms.RandomAffine(15, translate=(0.05,0.05), scale=(0.95, 1.05), shear=None, resample=0, fillcolor=0),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            # ImageNet normalization
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
        inp = self.train_transform(inp).contiguous()
        return inp


class ChestValTransforms:
    """
    Moco 2 modified  augmentation:
    Classic: https://arxiv.org/pdf/2003.04297.pdf
    Chest:   https://arxiv.org/pdf/2010.05352v1.pdf
    """

    def __init__(self, height: int = 128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            SquarePad(),
            transforms.Resize((height,height)),
            transforms.ToTensor(),
            # ImageNet normalization
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
        inp = self.train_transform(inp).contiguous()
        return inp