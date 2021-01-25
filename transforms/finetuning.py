from torchvision import transforms
import numpy as np

class ChestTrainTransforms:
    """
    Moco 2 modified  augmentation:
    Classic: https://arxiv.org/pdf/2003.04297.pdf
    Chest:   https://arxiv.org/pdf/2010.05352v1.pdf
    """

    def __init__(self, height: int = 128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.Resize((height,height)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
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