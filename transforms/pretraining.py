from torchvision import transforms
from pl_bolts.transforms.self_supervised import Patchify
import numpy as np

# MOCO

class Moco2TrainTransforms:
    """
    Moco 2 modified  augmentation:
    Classic: https://arxiv.org/pdf/2003.04297.pdf
    Chest:   https://arxiv.org/pdf/2010.05352v1.pdf
    """

    def __init__(self, height: int = 128):
        # image augmentation functions
        self.train_transform = transforms.Compose([
            transforms.Resize((height,height)),
            transforms.ColorJitter(brightness=0.45, contrast=0.45, saturation=0.45),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            # ImageNet normalization
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
        
        q = self.train_transform(inp).contiguous()
        k = self.train_transform(inp).contiguous()
        return q, k


class Moco2ValTransforms:
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
        
        q = self.train_transform(inp).contiguous()
        k = self.train_transform(inp).contiguous()
        return q, k    


#  CPCV2
class CPCTrainTransforms:
    """
    Transforms used for CPC:
    Transforms::
        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)
    Example::
        # in a regular dataset
        Imagenet(..., transforms=CPCTrainTransformsImageNet128())
        # in a DataModule
        module = ImagenetDataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCTrainTransformsImageNet128())
    """
    def __init__(self, patch_size: int = 32, overlap: int = 16):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        rand_crop = transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4), interpolation=3)
        col_jitter = transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)], p=0.8)

        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=self.patch_size, overlap_size=self.overlap),
        ])

        

        self.transforms = transforms.Compose([
            rand_crop,
            col_jitter,
            post_transform
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)

        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1


class CPCValTransforms:
    """
    Transforms used for CPC:
    Transforms::
        random_flip
        transforms.ToTensor()
        normalize
        Patchify(patch_size=patch_size, overlap_size=patch_size // 2)
    Example::
        # in a regular dataset
        Imagenet(..., transforms=CPCEvalTransformsImageNet128())
        # in a DataModule
        module = ImagenetDataModule(PATH)
        train_loader = module.train_dataloader(batch_size=32, transforms=CPCEvalTransformsImageNet128())
    """

    def __init__(self, patch_size: int = 32, overlap: int = 16):
        """
        Args:
            patch_size: size of patches when cutting up the image into overlapping patches
            overlap: how much to overlap patches
        """

        # image augmentation functions
        self.patch_size = patch_size
        self.overlap = overlap
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            Patchify(patch_size=patch_size, overlap_size=overlap),
        ])
        self.transforms = transforms.Compose([
            transforms.Resize(146, interpolation=3),
            transforms.CenterCrop(128),
            post_transform
        ])

    def __call__(self, inp):
        if isinstance(inp, np.ndarray):
            if len(inp.shape) == 3 and inp.shape[2] == 1:
                inp = np.dstack([inp, inp, inp])
            inp = transforms.ToPILImage()(inp)
            
        inp = self.flip_lr(inp)
        out1 = self.transforms(inp)
        return out1